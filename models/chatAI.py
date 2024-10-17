import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time


DEFAULT_SYSTEM_PROMPT = """
あなたは感情がなく、ヒアリングが得意なAIです。
あなたは患者の気づいていない医薬品への要望を聞き出せます。
質問の内容は以下の項目に準拠します。

1. 頭痛の頻度や持続時間はどのくらいですか？
2. どのような症状が伴いますか？
3. 頭痛の原因やきっかけは分かりますか？
4. 頭痛の程度はどのくらいですか？
5. 過去に頭痛で受診したことがありますか？
6. 現在、服用中の薬やサプリメントはありますか？
7. 薬を服用する際の制限や注意点はありますか？

注意: 医薬品や医療に関する質問に答えてはなりません。また、常に日本語で質問を投げかけてください。

"""


class chatAI_Lama3:
    def __init__(self, DEFAULT_SYSTEM_PROMPT=DEFAULT_SYSTEM_PROMPT ,\
                        model_name="elyza/Llama-3-ELYZA-JP-8B") -> None:

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
        )
        # self.model.eval()
        self.refresh(set_system=DEFAULT_SYSTEM_PROMPT)


    def send_message(self, text:str, max_new_tokens=250):
        self.count += 1
        self.messages.append({"role": "user", "content": text})
        token_ids = self._get_token_ids()
        reply = self._get_reply(token_ids, max_new_tokens)
        self.messages.append({"role": "assistant", "content": reply})
        
        # if self.count > 5:
        #     summary = self.summary_and_refresh()
        #     print(summary)
        return reply


    def _get_token_ids(self):        
        prompt = self.tokenizer.apply_chat_template(
            self.messages,
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
                temperature=0.1,
                top_p=0.9,
            )
        return self.tokenizer.decode(output_ids.tolist()[0][token_ids.size(1):], skip_special_tokens=True)
    
    def refresh(self, set_system=DEFAULT_SYSTEM_PROMPT):
        self.messages = [{"role": "system", "content": set_system}]
        self.count = 0


    def summary_and_refresh(self):
        content = ""
        for msg in self.messages:
            if msg["role"] == "assistant" or msg["role"] == "user":
                content += msg["role"] + "\n"
                content += msg["content"] + "\n"
        self.refresh(set_system="会話の内容を的確に要約します。日本語で答えます。")
        return self.send_message(text=content, max_new_tokens=500)
        


class similarityCritic:
    def __init__(self) -> None:
        # a = 'quora-distilbert-multilingual', "stsb-xlm-r-multilingual"
        self.model = SentenceTransformer('stsb-xlm-r-multilingual')

    def cos_sim(self, sentence :str, ref_sentence :str):
        embeddings1 = self.model.encode(sentence, convert_to_tensor=True)
        embeddings2 = self.model.encode(ref_sentence, convert_to_tensor=True)
        return util.pytorch_cos_sim(embeddings1, embeddings2)[0][0]



def main():
    chatAI = chatAI_Lama3()
    while True:
        user = input("メッセージを書いて> ")
        if user == "exit":
            break

        rep = chatAI.send_message(user)
        print(rep)


# if __name__ == "__main__":
#     main()



