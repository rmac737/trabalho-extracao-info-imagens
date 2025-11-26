import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = r"F:\github\trabalho-extracao-imagens-pucrj\Llama-3.1-8B-Instruct"

print("Transformers version:", __import__("transformers").__version__)
print("CUDA disponível?", torch.cuda.is_available())

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
)

dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    local_files_only=True,
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

if model.config.pad_token_id is None:
    model.config.pad_token_id = model.config.eos_token_id

messages = [
    {"role": "system", "content": "Você é um assistente técnico que responde em português."},
    {"role": "user", "content": "Explique de forma detalhada o que é uma API."},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(
    input_ids=input_ids,
    max_new_tokens=128,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)

generated = outputs[0, input_ids.shape[-1]:]
resposta = tokenizer.decode(generated, skip_special_tokens=True)

print("\n=== RESPOSTA ===\n")
print(resposta)