import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time

start = time.perf_counter()

def print_time(inp):
    now = time.perf_counter() 
    del_time = now-inp
    print(f'{del_time // 60}min {del_time % 60}sec')
    return now

DEFAULT_SYSTEM_PROMPT = "あなたは誠実で優秀な日本人のアシスタントです。特に指示が無い場合は、常に日本語で回答してください。"
text = "仕事の熱意を取り戻すためのアイデアを5つ挙げてください。"

model_name = "elyza/Llama-3-ELYZA-JP-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
quantization_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    quantization_config=quantization_config,
    low_cpu_mem_usage=True,
)
model.eval()

messages = [
    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
    {"role": "user", "content": text},
]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

token_ids = tokenizer.encode(
    prompt, add_special_tokens=False, return_tensors="pt"
)


bilde_time = print_time(start)

with torch.no_grad():
    output_ids = model.generate(
        token_ids.to(model.device),
        max_new_tokens=250,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
output = tokenizer.decode(output_ids.tolist()[0][token_ids.size(1):], skip_special_tokens=True)

print(output)

_ = print_time(bilde_time)
_ = print_time(start)
