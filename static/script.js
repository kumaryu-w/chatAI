
let wait = false;
document.getElementById('send-button').addEventListener('click', function() {

    const userInput = document.getElementById('user-input');
    const chatBox = document.getElementById('chat-box');

    if (userInput.value.trim() !== "" && !wait) {

        const userMessage = document.createElement('div');
        userMessage.classList.add('message', 'user');
        userMessage.innerHTML = `<p>${userInput.value}</p>`;
        chatBox.appendChild(userMessage);

        const userText = userInput.value
        userInput.value = '';
        
        // spnser
        const botMessage = document.createElement('div');
        botMessage.classList.add('message', 'bot');
        botMessage.innerHTML = `<div id="spinner" class="spinner"></div><p>しばらくお待ちください</p>`;
        chatBox.appendChild(botMessage);

        chatBox.scrollTop = chatBox.scrollHeight;

        wait = true;
        fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json',},
            body: JSON.stringify({'user_msg' : userText }),
        })
        .then(response => response.json())
        .then(data => {
            botMessage.innerHTML = `<p>${data.bot_msg}</p>`;
            chatBox.scrollTop = chatBox.scrollHeight;
            wait = false;
        })
        .catch((error) => {
            console.error('Error:', error);
            wait = false;
        });
    

    

    }

});
