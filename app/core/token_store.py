from datetime import datetime, timezone
from typing import Optional


class TokenStore:
    def __init__(self):
        # 블랙리스트: {jti: expire_timestamp}
        self._blacklist: dict[str, float] = {}
        # 활성 세션: {user_id: jti}
        self._active_sessions: dict[str, str] = {}

    # ── 블랙리스트 ──────────────────────────────

    def blacklist_token(self, jti: str, exp: float):
        """토큰을 블랙리스트에 추가 (exp: JWT exp 클레임 timestamp)"""
        self._blacklist[jti] = exp
        self._cleanup_blacklist()

    def is_blacklisted(self, jti: str) -> bool:
        return jti in self._blacklist

    def _cleanup_blacklist(self):
        """만료된 토큰은 자동 정리"""
        now = datetime.now(timezone.utc).timestamp()
        self._blacklist = {
            jti: exp for jti, exp in self._blacklist.items() if exp > now
        }

    # ── 활성 세션 ───────────────────────────────

    def set_active_session(self, user_id: str, jti: str):
        self._active_sessions[user_id] = jti

    def get_active_session(self, user_id: str) -> Optional[str]:
        return self._active_sessions.get(user_id)

    def remove_active_session(self, user_id: str):
        self._active_sessions.pop(user_id, None)

    def has_active_session(self, user_id: str) -> bool:
        return user_id in self._active_sessions


# 앱 전역 싱글톤
token_store = TokenStore()