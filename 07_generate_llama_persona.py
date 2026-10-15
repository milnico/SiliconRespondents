import torch
from transformers import AutoTokenizer
import argparse

from model.latent_llama import LatentLlamaModel

def generate_persona(model, tokenizer, device, traits, prompt_text):
    # traits should be [O, C, E, A, N]
    latents = torch.tensor(traits, dtype=torch.bfloat16).unsqueeze(0).to(device)
    
    messages = [
        {"role": "system", "content": "You are a human taking part in a psychological interview."},
        {"role": "user", "content": prompt_text}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    
    # We must set the condition before generating
    model.current_cond = model.latent_encoder(latents)
    
    with torch.no_grad():
        outputs = model.llm.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
    # Clear the condition
    model.current_cond = None
    
    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return generated_text

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = "unsloth/llama-3-8b-instruct-bnb-4bit"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = LatentLlamaModel(model_name=model_name, num_film_layers=6)
    
    state_dict = torch.load("checkpoints/llama_film_model.pt", map_location=device)
    filtered_dict = {k: v for k, v in state_dict.items() if "lora" in k or "latent_encoder" in k or "film_layers" in k}
    model.load_state_dict(filtered_dict, strict=False)
    
    model.latent_encoder.to(device)
    model.film_layers.to(device)
    model.eval()
    
    prompt = "Tell me a little bit about yourself, how do you usually spend your weekends?"
    
    print("\n=============================================")
    print("PROMPT:", prompt)
    print("=============================================\n")
    
    print(">>> Persona 1: Highly Extroverted, Low Neuroticism (O=0, C=0, E=2, A=0, N=-2)")
    text1 = generate_persona(model, tokenizer, device, [0.0, 0.0, 2.0, 0.0, -2.0], prompt)
    print(text1)
    print("-" * 50)
    
    print(">>> Persona 2: Highly Introverted, High Neuroticism (O=0, C=0, E=-2, A=0, N=2)")
    text2 = generate_persona(model, tokenizer, device, [0.0, 0.0, -2.0, 0.0, 2.0], prompt)
    print(text2)
    print("=============================================\n")

if __name__ == "__main__":
    main()
