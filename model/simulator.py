import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from .trait_encoder import TraitEncoder
from .gated_lora import LatentGatedLoRALinear


class RespondentSimulator(nn.Module):
    """Trait-conditioned generative respondent simulator.

    The Likert answer is produced through the backbone's own LM head (the
    next-token logits restricted to the digit tokens "1".."5"), so the same
    model retains its language ability and can also *generate text* in the
    voice of a trait profile.

    The 5-D latent (O,C,E,A,N) is injected through two high-bandwidth paths:
      1. A learned soft prefix prepended to the input embeddings (prefix tuning).
      2. Latent-gated LoRA on the attention projections of the final layers.
    """

    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct", target_layers=12,
                 num_prefix=8, trait_dim=256):
        super().__init__()

        print(f"Loading backbone: {model_name}...")
        self.causal_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            dtype=torch.bfloat16,
        )

        # Freeze the entire backbone (incl. LM head); we only train the
        # conditioning modules below.
        for param in self.causal_model.parameters():
            param.requires_grad = False

        self.config = self.causal_model.config
        self.hidden_dim = self.config.hidden_size
        self.num_prefix = num_prefix
        self.trait_dim = trait_dim

        # 5-D latent -> shared trait code, used by both conditioning paths.
        self.trait_encoder = TraitEncoder(in_dim=5, out_dim=trait_dim)
        # Trait code -> num_prefix soft tokens in embedding space.
        # Initialised small so training starts near the backbone's base behavior.
        self.prefix_proj = nn.Linear(trait_dim, num_prefix * self.hidden_dim)
        nn.init.normal_(self.prefix_proj.weight, std=0.02)
        nn.init.zeros_(self.prefix_proj.bias)

        self.target_layers = target_layers
        self.injected_modules = []
        self._inject_lora()

        # Token ids of the answer digits "1".."5" (set via set_answer_tokens).
        self.register_buffer("answer_token_ids", torch.arange(5), persistent=False)

        self.mode = "full"
        self.use_prefix = True

    @property
    def backbone(self):
        return self.causal_model.model

    def set_answer_tokens(self, tokenizer):
        ids = [tokenizer.convert_tokens_to_ids(str(d)) for d in range(1, 6)]
        assert all(i is not None and i >= 0 for i in ids), f"Bad digit token ids: {ids}"
        self.answer_token_ids = torch.tensor(ids, dtype=torch.long)

    def _inject_lora(self):
        """Wrap q/k/v/o projections in the final `target_layers` blocks."""
        layers = self.backbone.layers
        num_layers = len(layers)
        start_layer = num_layers - self.target_layers

        print(f"Injecting Latent-Gated LoRA into final {self.target_layers} layers "
              f"(q/k/v/o projections)...")
        for i in range(start_layer, num_layers):
            attn = layers[i].self_attn
            for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                base = getattr(attn, name)
                wrapped = LatentGatedLoRALinear(base, trait_dim=self.trait_dim)
                setattr(attn, name, wrapped)
                self.injected_modules.append(wrapped)

    def set_mode(self, mode):
        """
        mode: 'baseline_a' (no traits at all)
              'baseline_b' (traits in prompt text only, no prefix / no lora)
              'baseline_c' (soft prefix + LoRA, no gate)
              'full'       (soft prefix + latent-gated LoRA)
        """
        self.mode = mode
        use_lora = mode in ["baseline_c", "full"]
        use_gate = mode == "full"
        self.use_prefix = mode in ["baseline_c", "full"]

        for mod in self.injected_modules:
            mod.use_lora = use_lora
            mod.use_gate = use_gate

    def prepare_trainable(self, device, dtype=torch.bfloat16):
        """Move the trainable conditioning modules to device/dtype.

        The backbone is loaded separately via device_map="auto".
        """
        self.trait_encoder.to(device=device, dtype=dtype)
        self.prefix_proj.to(device=device, dtype=dtype)
        for mod in self.injected_modules:
            mod.lora_A.to(device=device, dtype=dtype)
            mod.lora_B.to(device=device, dtype=dtype)
            mod.gate_mlp.to(device=device, dtype=dtype)

    def trainable_parameters(self):
        params = list(self.trait_encoder.parameters()) + list(self.prefix_proj.parameters())
        for mod in self.injected_modules:
            params += list(mod.lora_A.parameters())
            params += list(mod.lora_B.parameters())
            params += list(mod.gate_mlp.parameters())
        return params

    def forward(self, input_ids, attention_mask, latent_traits=None):
        embed_tokens = self.backbone.embed_tokens
        tok_embeds = embed_tokens(input_ids)
        bsz = input_ids.size(0)

        # Compute the shared trait code and route it to the gated-LoRA modules.
        if latent_traits is not None and self.mode in ["baseline_c", "full"]:
            code = self.trait_encoder(latent_traits).to(tok_embeds.dtype)
            for mod in self.injected_modules:
                mod.current_trait_embedding = code
        else:
            code = None
            for mod in self.injected_modules:
                mod.current_trait_embedding = None

        # Soft-prefix conditioning path.
        if self.use_prefix and code is not None:
            prefix = self.prefix_proj(code).view(bsz, self.num_prefix, self.hidden_dim)
            prefix = prefix.to(tok_embeds.dtype)
            inputs_embeds = torch.cat([prefix, tok_embeds], dim=1)
            prefix_mask = torch.ones(bsz, self.num_prefix, dtype=attention_mask.dtype,
                                     device=attention_mask.device)
            attn_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        else:
            inputs_embeds = tok_embeds
            attn_mask = attention_mask

        # Robust position ids for left-padded sequences (+ prefix at front).
        position_ids = (attn_mask.long().cumsum(-1) - 1).clamp(min=0)
        position_ids = position_ids * attn_mask.long()

        outputs = self.causal_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            position_ids=position_ids,
        )

        # Logits for the final position; restrict to the 5 answer digits.
        last_logits = outputs.logits[:, -1, :]
        answer_ids = self.answer_token_ids.to(last_logits.device)
        digit_logits = last_logits.index_select(-1, answer_ids)

        return digit_logits.float()

    @torch.no_grad()
    def generate_text(self, input_ids, attention_mask, latent_traits, **gen_kwargs):
        """Free-text generation under a trait profile (the "speak like" path).

        The soft prefix and gated-LoRA condition the frozen LM head, so the
        generated continuation reflects the trait vector. Returns the generated
        token ids (continuation only, since `inputs_embeds` is used).
        """
        tok_embeds = self.backbone.embed_tokens(input_ids)
        bsz = input_ids.size(0)

        code = self.trait_encoder(latent_traits).to(tok_embeds.dtype)
        for mod in self.injected_modules:
            mod.current_trait_embedding = code if self.mode in ["baseline_c", "full"] else None

        inputs_embeds = tok_embeds
        attn_mask = attention_mask
        if self.use_prefix:
            prefix = self.prefix_proj(code).view(bsz, self.num_prefix, self.hidden_dim)
            inputs_embeds = torch.cat([prefix.to(tok_embeds.dtype), tok_embeds], dim=1)
            prefix_mask = torch.ones(bsz, self.num_prefix, dtype=attention_mask.dtype,
                                     device=attention_mask.device)
            attn_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        return self.causal_model.generate(
            inputs_embeds=inputs_embeds, attention_mask=attn_mask, **gen_kwargs)
