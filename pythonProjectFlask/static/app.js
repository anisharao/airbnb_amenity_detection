class Chatbox {
    constructor() {
        this.args = {
            openButton: document.querySelector('.chatbox__button'),
            chatBox: document.querySelector('.chatbox__support'),
            sendButton: document.querySelector('.send__button')
        }
        this.state = false;
        this.messages = [];
    }
    display(){
        const {openButton, chatBox, sendButton} = this.args;
        openButton.addEventListener('click',() => this.toggleState(chatBox))
        sendButton.addEventListener('click',()=> this.onSendButton(chatBox))
        const node = chatBox.querySelector('input');
        node.addEventListner("keyup",({key})=>{
            if(key === "Enter") {
                this.onSendButton(chatBox)
            }
        })
    }

    toggleState(chatbox) {
        this.state = !this.state;
        if(this.state){
            chatbox.classList.add('chatbox--active')
        } else{
            chatbox.classList.remove('chatbox--active')
        }
    }

    onSendButton(chatbox) {
        var textField = chatbox.querySelector('input');
        let text1 = textField.value
        if (text1 === ""){
            return;
        }

        let msg1 = {name: "User",message: text1}
        this.messages.push(msg1);

        fetch($SCRIPT_ROOT+ '/predict',{
            method: 'POST',
            body:JSON.stringify({message: text1}),
            mode:'cors',
            headers:{
                'Content-Type': 'application/json'
            },
        })
            .then(r => r.json())
            .then(r=>{
                let msg2 = { name:"sam", message: r.answer};
                this.messages.push(msg2);
                this.updateChatText(chatbox)
                textField.value = ''
            }).catch((error)=>{
                console.error('Error:',error);
                this.updateChatText(chatbox)
                textField.value = ''
            });

    }

    updateChatText(chatbox) {
        var html = "";
        console.log("petri",this.messages)
        this.messages = [
    {
        "name": "User",
        "message": "I'd like to get updates on my property inventory, please."
    },
    {
        "name": "sam",
        "message": "Certainly! Here's an update for your Property 1: You have 3 paper towels, 5 bowls, and 10 cups. Is there anything else you'd like to know? "
    }
]
        this.messages.slice().reverse().forEach(function(item,){
            if(item.name === "sam"){
                html += '<div class = "messages__item messages__item--visitor">' + item.message + '</div>'
            }
            else{
                html += '<div class="messages__item messages__item--operator">' + item.message + '</div>'
            }
        });

        const chatmessage = chatbox.querySelector('.chatbox__messages');
        chatmessage.innerHTML = html;
    }
}
const chatbox = new Chatbox();
chatbox.display()