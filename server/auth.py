"""演示级身份：账号、令牌、会话归属。

要解决的问题很具体：请求得知道「是谁发的」，会话得知道「属于谁」，
否则任何人换一个 ``thread_id`` 就能读到别人的对话历史。

不是生产鉴权——没有刷新令牌、没有角色权限、没有找回密码，账号是
``config/auth.yml`` 里的演示账号。密码用 PBKDF2-SHA256 加盐哈希存库，
令牌是 ``secrets.token_urlsafe`` 的随机串，带过期时间，都存在 SQLite 里，
重启不掉线。
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable

from utils.config_handler import auth_conf
from utils.path_tool import get_abs_path

_ALGORITHM = "pbkdf2_sha256"
_DEFAULT_ITERATIONS = 200_000
_SALT_BYTES = 16

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    display_name TEXT NOT NULL DEFAULT '',
    password_hash TEXT NOT NULL,
    business_user_id TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tokens (
    token TEXT PRIMARY KEY,
    username TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tokens_username ON tokens(username);

CREATE TABLE IF NOT EXISTS thread_owner (
    thread_id TEXT PRIMARY KEY,
    username TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_thread_owner_username ON thread_owner(username);
"""


@dataclass(frozen=True)
class User:
    username: str
    display_name: str = ""
    # 绑定的业务用户 ID（records.csv 里的 4 位号），报告链路用它拦越权查询。
    business_user_id: str | None = None

    def as_dict(self) -> dict:
        return {
            "username": self.username,
            "display_name": self.display_name or self.username,
            "user_id": self.business_user_id,
        }


# 鉴权关闭时的占位身份：接口不必到处判空。
ANONYMOUS_USER = User(username="anonymous", display_name="未登录")


def _iso(value: datetime) -> str:
    return value.isoformat(timespec="milliseconds")


def hash_password(
    password: str,
    *,
    iterations: int = _DEFAULT_ITERATIONS,
    salt: bytes | None = None,
) -> str:
    """返回 ``算法$迭代次数$盐$摘要``，四段都是可打印字符，便于直接存库。"""
    used_salt = salt or secrets.token_bytes(_SALT_BYTES)
    digest = hashlib.pbkdf2_hmac(
        "sha256", str(password).encode("utf-8"), used_salt, int(iterations)
    )
    return f"{_ALGORITHM}${int(iterations)}${used_salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """常量时间比较；格式不对一律当失败，不抛异常给调用方。"""
    parts = str(stored or "").split("$")
    if len(parts) != 4 or parts[0] != _ALGORITHM:
        return False
    try:
        iterations = int(parts[1])
        salt = bytes.fromhex(parts[2])
    except ValueError:
        return False
    digest = hashlib.pbkdf2_hmac(
        "sha256", str(password).encode("utf-8"), salt, iterations
    )
    return hmac.compare_digest(digest.hex(), parts[3])


class AuthStore:
    def __init__(
        self,
        db_path: str | None = None,
        *,
        enabled: bool | None = None,
        iterations: int | None = None,
        token_ttl_hours: float | None = None,
        demo_users: list[dict] | None = None,
        legacy_owner: str | None = None,
    ):
        conf = auth_conf or {}
        self.enabled = bool(conf.get("enabled", True)) if enabled is None else bool(enabled)
        self.iterations = int(
            iterations if iterations is not None else conf.get("pbkdf2_iterations", _DEFAULT_ITERATIONS)
        )
        self.token_ttl_hours = float(
            token_ttl_hours
            if token_ttl_hours is not None
            else conf.get("token_ttl_hours", 12)
        )
        configured_owner = (
            legacy_owner if legacy_owner is not None else conf.get("legacy_owner")
        )
        self.legacy_owner = str(configured_owner or "").strip() or None
        self.login_hint = str(conf.get("login_hint") or "")

        configured = db_path or conf.get("db_path", "checkpoints/auth.sqlite3")
        if not os.path.isabs(configured):
            configured = get_abs_path(configured)
        self.db_path = configured
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_schema()
        self._seed(demo_users if demo_users is not None else (conf.get("demo_users") or []))

    @contextmanager
    def _db(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._db() as conn:
            conn.executescript(_SCHEMA)

    def _seed(self, users: Iterable[dict]) -> None:
        """把配置里的演示账号写进去；已存在的账号不覆盖。

        环境变量 ``AUTH_DEMO_PASSWORD`` 优先于配置里的密码，方便在不改仓库
        文件的情况下换掉演示密码（只对新写入的账号生效；要改已有账号的密码，
        删掉 auth 库重启即可）。
        """
        override = os.getenv("AUTH_DEMO_PASSWORD", "").strip()
        wanted = []
        for item in users or []:
            username = str((item or {}).get("username") or "").strip()
            password = override or str((item or {}).get("password") or "")
            if not username or not password:
                continue
            wanted.append(
                (
                    username,
                    str((item or {}).get("display_name") or username),
                    password,
                    str((item or {}).get("business_user_id") or "").strip() or None,
                )
            )
        if not wanted:
            return

        with self._db() as conn:
            existing = {
                str(row["username"])
                for row in conn.execute(
                    "SELECT username FROM users WHERE username IN (%s)"
                    % ",".join("?" * len(wanted)),
                    tuple(name for name, _, _, _ in wanted),
                ).fetchall()
            }
            for username, display_name, password, business_user_id in wanted:
                if username in existing:
                    # PBKDF2 一次 0.07s，重启时只为新账号付这个代价。
                    continue
                conn.execute(
                    """
                    INSERT OR IGNORE INTO users
                        (username, display_name, password_hash, business_user_id, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        username,
                        display_name,
                        hash_password(password, iterations=self.iterations),
                        business_user_id,
                        _iso(datetime.now(timezone.utc)),
                    ),
                )

    # ---------------------------------------------------------------- 账号

    @staticmethod
    def _user_from_row(row: sqlite3.Row) -> User:
        return User(
            username=str(row["username"]),
            display_name=str(row["display_name"] or ""),
            business_user_id=row["business_user_id"],
        )

    def demo_users(self) -> list[dict]:
        """只给前端展示用的字段，密码哈希绝不出库。"""
        with self._db() as conn:
            rows = conn.execute(
                "SELECT username, display_name, business_user_id FROM users ORDER BY rowid"
            ).fetchall()
        return [
            {
                "username": str(row["username"]),
                "display_name": str(row["display_name"] or row["username"]),
                "user_id": row["business_user_id"],
            }
            for row in rows
        ]

    def login(self, username: str, password: str) -> tuple[User, str] | None:
        username = str(username or "").strip()
        if not username or not password:
            return None
        now = datetime.now(timezone.utc)
        with self._db() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE username = ?", (username,)
            ).fetchone()
            if row is None or not verify_password(password, row["password_hash"]):
                return None
            user = self._user_from_row(row)
            token = secrets.token_urlsafe(32)
            conn.execute(
                "INSERT INTO tokens (token, username, created_at, expires_at) VALUES (?, ?, ?, ?)",
                (
                    token,
                    user.username,
                    _iso(now),
                    _iso(now + timedelta(hours=self.token_ttl_hours)),
                ),
            )
            # 顺手清掉这个用户的过期令牌，避免表随登录次数无限增长。
            conn.execute(
                "DELETE FROM tokens WHERE username = ? AND expires_at < ?",
                (user.username, _iso(now)),
            )
        return user, token

    def logout(self, token: str) -> None:
        token = str(token or "").strip()
        if not token:
            return
        with self._db() as conn:
            conn.execute("DELETE FROM tokens WHERE token = ?", (token,))

    def user_by_token(self, token: str | None) -> User | None:
        token = str(token or "").strip()
        if not token:
            return None
        now = _iso(datetime.now(timezone.utc))
        with self._db() as conn:
            row = conn.execute(
                """
                SELECT users.* FROM tokens
                JOIN users ON users.username = tokens.username
                WHERE tokens.token = ? AND tokens.expires_at >= ?
                """,
                (token, now),
            ).fetchone()
            if row is None:
                # 过期令牌直接删，避免表里堆积永远用不上的行。
                conn.execute(
                    "DELETE FROM tokens WHERE token = ? AND expires_at < ?", (token, now)
                )
                return None
            return self._user_from_row(row)

    # ------------------------------------------------------------ 会话归属

    def thread_owner(self, thread_id: str) -> str | None:
        thread_id = str(thread_id or "").strip()
        if not thread_id:
            return None
        with self._db() as conn:
            row = conn.execute(
                "SELECT username FROM thread_owner WHERE thread_id = ?", (thread_id,)
            ).fetchone()
        return str(row["username"]) if row else None

    def claim_thread(self, thread_id: str, username: str) -> str | None:
        """把会话归到 ``username`` 名下；已被别人占用时返回真正的归属者。"""
        thread_id = str(thread_id or "").strip()
        if not thread_id:
            return None
        with self._db() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO thread_owner (thread_id, username, created_at) VALUES (?, ?, ?)",
                (thread_id, username, _iso(datetime.now(timezone.utc))),
            )
            row = conn.execute(
                "SELECT username FROM thread_owner WHERE thread_id = ?", (thread_id,)
            ).fetchone()
        return str(row["username"]) if row else None

    def release_thread(self, thread_id: str) -> None:
        with self._db() as conn:
            conn.execute(
                "DELETE FROM thread_owner WHERE thread_id = ?", (str(thread_id or ""),)
            )

    def owned_thread_ids(self, username: str) -> set[str]:
        with self._db() as conn:
            rows = conn.execute(
                "SELECT thread_id FROM thread_owner WHERE username = ?", (username,)
            ).fetchall()
        return {str(row["thread_id"]) for row in rows}

    def owner_filter(self, username: str) -> Callable[[str], bool]:
        """给 ``list_sessions`` 用的谓词：先取一次归属集合，避免逐条查库。"""
        owned = self.owned_thread_ids(username)
        return lambda thread_id: str(thread_id) in owned

    def adopt_threads(self, thread_ids: Iterable[str], username: str) -> int:
        """把还没有归属的旧会话一次性归到 ``username``，返回新认领的数量。"""
        username = str(username or "").strip()
        if not username:
            return 0
        adopted = 0
        now = _iso(datetime.now(timezone.utc))
        with self._db() as conn:
            for thread_id in thread_ids or []:
                value = str(thread_id or "").strip()
                if not value:
                    continue
                cursor = conn.execute(
                    "INSERT OR IGNORE INTO thread_owner (thread_id, username, created_at) VALUES (?, ?, ?)",
                    (value, username, now),
                )
                adopted += cursor.rowcount or 0
        return adopted


__all__ = [
    "ANONYMOUS_USER",
    "AuthStore",
    "User",
    "hash_password",
    "verify_password",
]
