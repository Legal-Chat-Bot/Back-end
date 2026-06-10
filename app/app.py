from fastapi import FastAPI, Depends
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.db import get_db, init_db
from app.routes.auth.login import router as login_router
from app.routes.auth.signup import router as signup_router
from app.routes.user.user import router as user_router
from app.routes.websocket.chat import router as chat_router
import os

app = FastAPI(title=settings.PROJECT_NAME)
app.include_router(login_router)
app.include_router(signup_router)
app.include_router(user_router)
app.include_router(chat_router)

init_db()  # 애플리케이션 시작 시 DB 초기화 (테이블 생성 등)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

html = """
<!DOCTYPE html>
<html>
    <head><title>Chat</title></head>
    <body>
        <h1>WebSocket Chat</h1>
        <div>
            <input type="text" id="tokenInput" placeholder="토큰 입력" style="width:400px"/>
            <br/><br/>
            <input type="text" id="sessionInput" placeholder="session_id 입력 (새 대화면 비워두세요)" style="width:400px"/>
            <br/><br/>
            <button onclick="connect()">연결</button>
        </div>
        <br/>
        <form action="" onsubmit="sendMessage(event)">
            <input type="text" id="messageText" autocomplete="off"/>
            <button>Send</button>
        </form>
        <ul id='messages'></ul>
        <script>
            var ws;

            function connect() {
                var token = document.getElementById("tokenInput").value;
                var session_id = document.getElementById("sessionInput").value;

                var url = `ws://localhost:8000/ws/chat?token=${token}`;
                if (session_id) {
                    url += `&session_id=${session_id}`;
                }

                ws = new WebSocket(url);

                ws.onmessage = function(event) {
                    var data = JSON.parse(event.data);
                    var messages = document.getElementById('messages');
                    var message = document.createElement('li');

                    if (data.type === "session_created") {
                        document.getElementById("sessionInput").value = data.session_id;
                        message.textContent = `[세션 생성됨] session_id: ${data.session_id}`;
                    } else if (data.type === "error") {
                        message.textContent = `[에러] ${data.message}`;
                    } else {
                        message.textContent = data.content;
                    }

                    messages.appendChild(message);
                };

                ws.onopen = function() {
                    var messages = document.getElementById('messages');
                    var message = document.createElement('li');
                    message.textContent = "[연결 성공]";
                    messages.appendChild(message);
                };

                ws.onclose = function(e) {
                    var messages = document.getElementById('messages');
                    var message = document.createElement('li');
                    message.textContent = `[연결 종료] code: ${e.code}`;
                    messages.appendChild(message);
                };
            }

            function sendMessage(event) {
                if (!ws) {
                    alert("먼저 연결하세요.");
                    event.preventDefault();
                    return;
                }
                var input = document.getElementById("messageText");
                ws.send(JSON.stringify({ message: input.value }));
                input.value = '';
                event.preventDefault();
            }
        </script>
    </body>
</html>
"""

@app.get("/")
async def get():
    return HTMLResponse(html)

@app.get("/test")
def read_test(
    db: Session = Depends(get_db)
):
    return {"message": "Hello, World!"}