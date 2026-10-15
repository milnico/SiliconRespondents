import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from model.latent_encoder import LatentEncoder
from model.film_layers import FiLMLayer

class LatentLlamaModel(nn.Module):
    def __init__(self, model_name="unsloth/llama-3-8b-instruct-bnb-4bit", num_film_layers=6):
        super().__init__()
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.llm.gradient_checkpointing_enable()
        self.llm.enable_input_require_grads()
        self.llm = get_peft_model(self.llm, peft_config)
        
        hidden_dim = self.llm.config.hidden_size
        cond_dim = 256
        self.latent_encoder = LatentEncoder(in_dim=5, out_dim=cond_dim).to(torch.bfloat16)
        
        self.num_film_layers = num_film_layers
        self.film_layers = nn.ModuleList([
            FiLMLayer(cond_dim=cond_dim, hidden_dim=hidden_dim).to(torch.bfloat16)
            for _ in range(num_film_layers)
        ])
        
        self.num_hidden_layers = self.llm.config.num_hidden_layers
        self.current_cond = None
        self._register_hooks()
        
    def _register_hooks(self):
        start_layer = self.num_hidden_layers - self.num_film_layers
        
        def make_hook(layer_idx):
            def hook(module, args, output):
                is_tuple = isinstance(output, tuple)
                hidden_states = output[0] if is_tuple else output
                
                if self.current_cond is not None:
                    film_idx = layer_idx - start_layer
                    film_layer = self.film_layers[film_idx]
                    
                    layer_dtype = film_layer.gamma.weight.dtype
                    orig_dtype = hidden_states.dtype
                    cond = self.current_cond.to(layer_dtype)
                    hidden_states = film_layer(hidden_states.to(layer_dtype), cond).to(orig_dtype)
                    
                if is_tuple:
                    return (hidden_states,) + output[1:]
                return hidden_states
            return hook
            
        for i in range(start_layer, self.num_hidden_layers):
            layer = self.llm.base_model.model.model.layers[i]
            layer.register_forward_hook(make_hook(i))
            
    def forward(self, input_ids, attention_mask, latent_traits, labels=None):
        latent_traits = latent_traits.to(next(self.latent_encoder.parameters()).dtype)
        self.current_cond = self.latent_encoder(latent_traits)
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        self.current_cond = None
        return outputs
