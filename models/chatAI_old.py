import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time


def print_time(inp):
    now = time.perf_counter() 
    del_time = now-inp
    print(f'{del_time // 60}min {del_time % 60}sec')
    return now


class chatAI_Lama3:
    def __init__(self, DEFAULT_SYSTEM_PROMPT="あなたは誠実で優秀な日本人のアシスタントです。特に指示が無い場合は、常に日本語で回答してください。" ,\
                        model_name="elyza/Llama-3-ELYZA-JP-8B") -> None:

        self.DEFAULT_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
        )
        self.model.eval()

    def send_message(self, text:str, max_new_tokens=250):
        token_ids = self._get_token_ids(text=text)
        reply = self._get_reply(token_ids, max_new_tokens)
        return reply


    def _get_token_ids(self, text:str):        
        messages = [
            {"role": "system", "content": self.DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": text},]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        token_ids = self.tokenizer.encode(
            prompt, add_special_tokens=False, return_tensors="pt"
        )
        return token_ids
    
    def _get_reply(self, token_ids, max_new_tokens):
        with torch.no_grad():
            output_ids = self.model.generate(
                token_ids.to(self.model.device),
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )
        reply = self.tokenizer.decode(output_ids.tolist()[0][token_ids.size(1):], skip_special_tokens=True)
        return reply
        

def main():
    chatAI = chatAI_Lama3()
    while True:
        user = input("メッセージを書いて> ")
        if user == "exit":
            break

        rep = chatAI.send_message(user)
        print(rep)


if __name__ == "__main__":
    main()

