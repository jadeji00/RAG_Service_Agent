"""按用户的滑动窗口限流。

要解决的问题：``/chat/stream`` 每发一次都要调真实模型（主模型 + 可能的重排和
摘要），一个写错的循环脚本或者被贴出去的演示地址就能把配额烧光。

为什么是滑动窗口而不是固定窗口：固定窗口在两个窗口的交界处能放过 2 倍流量
（第 59 秒打满、第 61 秒再打满）。滑动窗口只记最近 ``window_seconds`` 内的
命中时刻，任何 60 秒区间里都不会超额。

为什么是进程内存而不是 Redis：这是单进程的演示服务，引入 Redis 意味着多一个
要装、要起、要连不上的东西。**代价是限流按 worker 各自计数、重启即清零**，
多 worker 部署时实际上限是 ``max_requests × worker 数``——这一点写在
``docs/design.md`` 的取舍表里，不是隐藏的。
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

from utils.config_handler import auth_conf

_DEFAULT_WINDOW_SECONDS = 60.0
_DEFAULT_MAX_REQUESTS = 20
# 跟踪的 key 数量上限。超过就清一遍已经空了的 key，防止长时间运行后
# 字典里堆满早就走掉的客户端。
_MAX_TRACKED_KEYS = 4096


@dataclass(frozen=True)
class LimitDecision:
    """一次限流判定的结果。

    ``retry_after`` 只在被拒时有意义：等到最老的那次命中滑出窗口还需要多少秒，
    直接拿去当 HTTP 的 ``Retry-After`` 头。
    """

    allowed: bool
    retry_after: int = 0
    remaining: int = 0
    limit: int = 0


class SlidingWindowLimiter:
    """每个 key 一条命中时刻队列，判定前先剪掉窗口外的。"""

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        window_seconds: float | None = None,
        max_requests: int | None = None,
        clock: Callable[[], float] | None = None,
    ):
        conf = (auth_conf or {}).get("rate_limit") or {}
        self.enabled = bool(conf.get("enabled", True)) if enabled is None else bool(enabled)
        self.window_seconds = float(
            window_seconds
            if window_seconds is not None
            else conf.get("window_seconds", _DEFAULT_WINDOW_SECONDS)
        )
        self.max_requests = int(
            max_requests
            if max_requests is not None
            else conf.get("max_requests", _DEFAULT_MAX_REQUESTS)
        )
        # monotonic 而不是 time()：系统时钟被往回调时，time() 会让窗口
        # 「变长」甚至永久卡住；monotonic 只会往前走。
        self._clock = clock or time.monotonic
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def hit(self, key: str) -> LimitDecision:
        """记一次命中并返回判定。被拒时**不记**这一次。

        被拒也记的话，客户端每重试一次就把惩罚往后推一次，一个死循环的脚本
        会把自己永久锁死；不记则 ``retry_after`` 是确定的，窗口一滑出来就恢复。
        """
        if not self.enabled or self.max_requests <= 0:
            return LimitDecision(True, remaining=self.max_requests, limit=self.max_requests)

        now = self._clock()
        edge = now - self.window_seconds
        with self._lock:
            hits = self._hits.get(key)
            if hits is None:
                hits = deque()
                self._hits[key] = hits
            while hits and hits[0] <= edge:
                hits.popleft()
            if len(hits) >= self.max_requests:
                # 最老那次滑出窗口的时刻，就是最早可以重试的时刻。
                retry_after = max(1, math.ceil(hits[0] - edge))
                return LimitDecision(
                    False, retry_after=retry_after, remaining=0, limit=self.max_requests
                )
            hits.append(now)
            if len(self._hits) > _MAX_TRACKED_KEYS:
                self._sweep(edge)
            return LimitDecision(
                True,
                remaining=self.max_requests - len(hits),
                limit=self.max_requests,
            )

    def _sweep(self, edge: float) -> None:
        """丢掉窗口内已经没有命中的 key；调用方需持有锁。"""
        for key in [k for k, hits in self._hits.items() if not hits or hits[-1] <= edge]:
            self._hits.pop(key, None)

    def reset(self, key: str | None = None) -> None:
        """清掉一个 key（或全部）的计数；单测和账号注销时用。"""
        with self._lock:
            if key is None:
                self._hits.clear()
            else:
                self._hits.pop(key, None)


__all__ = ["LimitDecision", "SlidingWindowLimiter"]
