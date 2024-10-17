from flask import Flask, request, jsonify, render_template
from models.chatAI import chatAI_Lama3, similarityCritic

app = Flask(__name__)
CHAT_AI = chatAI_Lama3()
SC = similarityCritic()

@app.route('/')
def index():
    CHAT_AI.refresh()
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_msg = request.json.get("user_msg")
    if not user_msg:
        return jsonify({"bot_msg": "通信エラーが発生しました。\n再度読み込みをしてみてください"}), 400
    
    response = CHAT_AI.send_message(user_msg)
    
    if len(CHAT_AI.messages) >= 3:
        s1 = CHAT_AI.messages[-3]["content"]
        s2 = "薬を服用する際の制限や注意点はありますか？"
        if SC.cos_sim(s1, s2) >= 0.9:
            summary = CHAT_AI.summary_and_refresh()
            print(summary)



    return jsonify({"bot_msg": response})


if __name__ == '__main__':
    app.run(debug=True)

