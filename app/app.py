from fastapi import FastAPI, Depends
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.db import get_db, init_db
from app.routes.auth.login import router as login_router
from app.routes.auth.signup import router as signup_router
from app.routes.user.user import router as user_router
from app.routes.websocket.chat import router as websocket_router
from app.routes.chat.session import router as chat_router
from app.routes.chat.document import router as document_router
from app.routes.oauth.kakao import router as kakao_router
import os

app = FastAPI(title=settings.PROJECT_NAME)
app.include_router(login_router)
app.include_router(signup_router)
app.include_router(user_router)
app.include_router(chat_router)
app.include_router(document_router)
app.include_router(kakao_router)
app.include_router(websocket_router)

init_db()  # 애플리케이션 시작 시 DB 초기화 (테이블 생성 등)

html="""
<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>카카오 OAuth 테스트</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: 'Segoe UI', sans-serif;
      background: #0f0f0f;
      color: #e0e0e0;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 24px;
    }

    .container {
      width: 100%;
      max-width: 560px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    h1 {
      font-size: 18px;
      font-weight: 600;
      color: #fff;
      margin-bottom: 4px;
    }

    .subtitle {
      font-size: 13px;
      color: #666;
      margin-bottom: 8px;
    }

    /* 단계 카드 */
    .card {
      background: #1a1a1a;
      border: 1px solid #2a2a2a;
      border-radius: 12px;
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .card-header {
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .step-badge {
      background: #2a2a2a;
      color: #888;
      font-size: 11px;
      font-weight: 700;
      padding: 3px 8px;
      border-radius: 20px;
      letter-spacing: 0.5px;
    }

    .step-badge.active { background: #FEE500; color: #1a1a1a; }
    .step-badge.done   { background: #1f3a1f; color: #4caf50; }

    .card-title {
      font-size: 14px;
      font-weight: 600;
      color: #ccc;
    }

    /* 버튼 */
    .btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      padding: 11px 20px;
      border-radius: 8px;
      border: none;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
      transition: opacity 0.15s;
      text-decoration: none;
    }
    .btn:hover { opacity: 0.85; }
    .btn:disabled { opacity: 0.4; cursor: not-allowed; }

    .btn-kakao  { background: #FEE500; color: #1a1a1a; width: 100%; }
    .btn-blue   { background: #1d6bf3; color: #fff; width: 100%; }
    .btn-ghost  { background: #2a2a2a; color: #ccc; width: 100%; }

    /* 인풋 */
    .input-wrap { display: flex; gap: 8px; }
    input[type="text"] {
      flex: 1;
      background: #111;
      border: 1px solid #2a2a2a;
      border-radius: 8px;
      padding: 10px 14px;
      font-size: 13px;
      color: #e0e0e0;
      outline: none;
      transition: border-color 0.15s;
    }
    input[type="text"]:focus { border-color: #444; }
    input[type="text"]::placeholder { color: #444; }

    /* 결과 박스 */
    .result-box {
      background: #111;
      border: 1px solid #2a2a2a;
      border-radius: 8px;
      padding: 14px;
      font-size: 12px;
      font-family: 'Courier New', monospace;
      color: #4caf50;
      white-space: pre-wrap;
      word-break: break-all;
      min-height: 60px;
      display: none;
    }
    .result-box.error { color: #f44336; }
    .result-box.show  { display: block; }

    /* URL 박스 */
    .url-box {
      background: #111;
      border: 1px solid #2a2a2a;
      border-radius: 8px;
      padding: 12px 14px;
      font-size: 12px;
      color: #888;
      word-break: break-all;
      display: none;
    }
    .url-box.show { display: block; }

    .hint {
      font-size: 12px;
      color: #555;
      line-height: 1.6;
    }
    .hint b { color: #888; }

    /* 구분선 */
    hr { border: none; border-top: 1px solid #1f1f1f; }

    /* 토큰 표시 */
    .token-item {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .token-label { font-size: 11px; color: #555; }
    .token-value {
      font-size: 11px;
      font-family: monospace;
      color: #4caf50;
      word-break: break-all;
      background: #111;
      padding: 8px 10px;
      border-radius: 6px;
      border: 1px solid #1f1f1f;
    }

    .tag-new  { font-size: 11px; color: #FEE500; margin-left: 6px; }
    .tag-exist{ font-size: 11px; color: #4caf50; margin-left: 6px; }
  </style>
</head>
<body>
<div class="container">

  <div>
    <h1>🧪 카카오 OAuth 테스트</h1>
    <p class="subtitle">백엔드 자체 테스트용 페이지 — 순서대로 진행하세요</p>
  </div>

  <!-- STEP 1 : 로그인 URL 받기 -->
  <div class="card" id="card1">
    <div class="card-header">
      <span class="step-badge active" id="badge1">STEP 1</span>
      <span class="card-title">카카오 로그인 URL 받기</span>
    </div>
    <p class="hint">백엔드에서 <b>GET /oauth/kakao/login-url</b> 을 호출해 카카오 로그인 페이지 URL을 받아옵니다.</p>
    <button class="btn btn-kakao" onclick="getLoginUrl()">
      🔑 로그인 URL 받기
    </button>
    <div class="url-box" id="loginUrlBox"></div>
    <div class="result-box" id="result1"></div>
    <!-- URL 받은 후 카카오로 이동 버튼 -->
    <button class="btn btn-ghost" id="goKakaoBtn" style="display:none" onclick="goKakao()">
      카카오 로그인 페이지로 이동 →
    </button>
  </div>

  <!-- STEP 2 : 인가 코드 입력 -->
  <div class="card" id="card2">
    <div class="card-header">
      <span class="step-badge" id="badge2">STEP 2</span>
      <span class="card-title">인가 코드 입력</span>
    </div>
    <p class="hint">
      카카오 로그인 완료 후 브라우저 주소창에서 <b>?code=</b> 뒤의 값을 복사해 붙여넣으세요.<br>
      예: <b>http://localhost:5173/auth/kakao/callback?code=<span style="color:#FEE500">여기값</span></b>
    </p>
    <div class="input-wrap">
      <input type="text" id="codeInput" placeholder="카카오 인가 코드를 여기에 붙여넣으세요" />
      <button class="btn btn-blue" style="width:auto; padding: 10px 16px;" onclick="submitCode()">전송</button>
    </div>
    <div class="result-box" id="result2"></div>
  </div>

  <!-- STEP 3 : 결과 -->
  <div class="card" id="card3" style="display:none">
    <div class="card-header">
      <span class="step-badge done" id="badge3">DONE</span>
      <span class="card-title">JWT 발급 완료 <span id="newUserTag"></span></span>
    </div>
    <div class="token-item">
      <span class="token-label">ACCESS TOKEN</span>
      <div class="token-value" id="accessTokenVal">—</div>
    </div>
    <div class="token-item">
      <span class="token-label">REFRESH TOKEN</span>
      <div class="token-value" id="refreshTokenVal">—</div>
    </div>
    <hr />
    <button class="btn btn-ghost" onclick="reset()">↺ 처음부터 다시 테스트</button>
  </div>

</div>

<script>
  const API = 'http://localhost:8000'; // 백엔드 주소
  let kakaoLoginUrl = '';

  // STEP 1 - 로그인 URL 받기
  async function getLoginUrl() {
    const result1 = document.getElementById('result1');
    const urlBox  = document.getElementById('loginUrlBox');

    try {
      const res  = await fetch(`${API}/auth/kakao/login-url`);
      const data = await res.json();

      kakaoLoginUrl = data.login_url;

      // URL 표시
      urlBox.textContent = kakaoLoginUrl;
      urlBox.classList.add('show');

      // 카카오 이동 버튼 표시
      document.getElementById('goKakaoBtn').style.display = 'flex';

      // badge 완료 표시
      document.getElementById('badge1').className = 'step-badge done';
      document.getElementById('badge2').className = 'step-badge active';

      result1.className = 'result-box';
    } catch (e) {
      result1.textContent = `❌ 오류: 백엔드 서버가 실행 중인지 확인하세요.\n${e}`;
      result1.className = 'result-box error show';
    }
  }

  // STEP 1 - 카카오 로그인 페이지로 이동
  function goKakao() {
    if (!kakaoLoginUrl) return;
    // 새 탭에서 열기 (콜백 후 code를 복사해서 돌아올 수 있게)
    window.open(kakaoLoginUrl, '_blank');
  }

  // STEP 2 - 인가 코드 → JWT 발급
  async function submitCode() {
    const code    = document.getElementById('codeInput').value.trim();
    const result2 = document.getElementById('result2');

    if (!code) {
      result2.textContent = '⚠️ 인가 코드를 입력해주세요.';
      result2.className = 'result-box error show';
      return;
    }

    result2.textContent = '⏳ 처리 중...';
    result2.className = 'result-box show';

    try {
      const res  = await fetch(`${API}/auth/kakao/callback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code }),
      });

      const data = await res.json();

      if (!res.ok) {
        result2.textContent = `❌ 오류 ${res.status}\n${JSON.stringify(data, null, 2)}`;
        result2.className = 'result-box error show';
        return;
      }

      // 성공 → STEP 3 표시
      result2.className = 'result-box';
      showResult(data);

    } catch (e) {
      result2.textContent = `❌ 네트워크 오류\n${e}`;
      result2.className = 'result-box error show';
    }
  }

  // STEP 3 - 결과 표시
  function showResult(data) {
    document.getElementById('card3').style.display = 'flex';
    document.getElementById('badge2').className = 'step-badge done';

    document.getElementById('accessTokenVal').textContent  = data.access_token  || '—';
    document.getElementById('refreshTokenVal').textContent = data.refresh_token || '(refresh_token 없음)';

    const tag = document.getElementById('newUserTag');
    if (data.is_new_user) {
      tag.textContent = '🆕 신규 회원가입';
      tag.className = 'tag-new';
    } else {
      tag.textContent = '✅ 기존 유저 로그인';
      tag.className = 'tag-exist';
    }

    // 카드3으로 스크롤
    document.getElementById('card3').scrollIntoView({ behavior: 'smooth' });
  }

  // 초기화
  function reset() {
    kakaoLoginUrl = '';
    document.getElementById('codeInput').value = '';
    document.getElementById('loginUrlBox').classList.remove('show');
    document.getElementById('goKakaoBtn').style.display = 'none';
    document.getElementById('result1').className = 'result-box';
    document.getElementById('result2').className = 'result-box';
    document.getElementById('card3').style.display = 'none';
    document.getElementById('badge1').className = 'step-badge active';
    document.getElementById('badge2').className = 'step-badge';
    document.getElementById('accessTokenVal').textContent  = '—';
    document.getElementById('refreshTokenVal').textContent = '—';
  }

  // 엔터키로 코드 전송
  document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('codeInput').addEventListener('keydown', (e) => {
      if (e.key === 'Enter') submitCode();
    });
  });
</script>
</body>
</html>
"""



app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def get():
    return html

@app.get("/test")
def read_test(
    db: Session = Depends(get_db)
):
    return {"message": "Hello, World!"}


