from flask import Flask, request, jsonify, render_template_string
from project.rag.pipeline import answer_query

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Tutor Chat</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f5f6fa;
            padding: 30px;
        }
        h1 {
            margin-bottom: 15px;
            font-size: 32px;
            color: #2c3e50;
        }

        #chat-box {
            width: 90%;
            height: 550px;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #ccc;
            background: white;
            overflow-y: scroll;
            margin-bottom: 10px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        }

        .msg-user {
            color: #1e40af;
            margin: 8px 0;
            font-weight: bold;
        }

        .msg-ai {
            color: #222;
            margin: 8px 0;
            padding: 10px;
            background: #eef1f7;
            border-radius: 6px;
        }

        #input-box {
            width: 75%;
            padding: 10px;
            font-size: 16px;
            border-radius: 6px;
            border: 1px solid #aaa;
        }

        #send-btn {
            padding: 10px 18px;
            font-size: 16px;
            border-radius: 6px;
            background: #2ecc71;
            color: white;
            border: none;
            cursor: pointer;
        }

        #send-btn:hover {
            background: #27ae60;
        }

        /* Loading animation */
        #loading {
            display: none;
            font-style: italic;
            color: #888;
            margin-top: 5px;
        }

        .dot {
            animation: blink 1.4s infinite both;
        }
        .dot:nth-child(2) {
            animation-delay: .2s;
        }
        .dot:nth-child(3) {
            animation-delay: .4s;
        }
        @keyframes blink {
            0% { opacity: .2; }
            20% { opacity: 1; }
            100% { opacity: .2; }
        }
    </style>
</head>

<body>
    <h1>AI Tutor Chat</h1>

    <div id="chat-box"></div>

    <input id="input-box" type="text" placeholder="Ask a question...">
    <button id="send-btn" onclick="sendMessage()">Send</button>

    <div id="loading">AI thinking<span class="dot">.</span><span class="dot">.</span><span class="dot">.</span></div>

<script>
function addMessage(text, cls) {
    const box = document.getElementById("chat-box");
    const div = document.createElement("div");
    div.className = cls;
    div.innerHTML = text.replace(/\\n/g, "<br>");
    box.appendChild(div);
    box.scrollTop = box.scrollHeight;
}

async function sendMessage() {
    const input = document.getElementById("input-box");
    const msg = input.value.trim();
    if (!msg) return;

    addMessage("You: " + msg, "msg-user");
    input.value = "";

    document.getElementById("loading").style.display = "inline";

    const res = await fetch("/ask", {
        method: "POST",
        body: msg
    });
    const data = await res.json();

    document.getElementById("loading").style.display = "none";
    addMessage(data.answer, "msg-ai");
}
</script>

</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/ask", methods=["POST"])
def ask():
    query = request.data.decode("utf-8")
    response = answer_query(query)
    return jsonify({"answer": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
