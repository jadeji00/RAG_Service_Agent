import unittest

from server.rate_limit import SlidingWindowLimiter


class FakeClock:
    """可手动推进的时钟：限流逻辑全靠时间，用真时钟测就得真的等。"""

    def __init__(self, start: float = 0.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class SlidingWindowLimiterTest(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.limiter = SlidingWindowLimiter(
            enabled=True, window_seconds=60, max_requests=3, clock=self.clock
        )

    def test_allows_up_to_limit_then_rejects(self):
        decisions = [self.limiter.hit("demo") for _ in range(4)]

        self.assertEqual([d.allowed for d in decisions], [True, True, True, False])
        self.assertEqual([d.remaining for d in decisions[:3]], [2, 1, 0])
        self.assertEqual(decisions[3].limit, 3)
        self.assertGreaterEqual(decisions[3].retry_after, 1)

    def test_window_slides_and_budget_comes_back(self):
        for _ in range(3):
            self.limiter.hit("demo")
        self.assertFalse(self.limiter.hit("demo").allowed)

        self.clock.advance(61)

        self.assertTrue(self.limiter.hit("demo").allowed)

    def test_retry_after_is_time_until_oldest_hit_expires(self):
        limiter = SlidingWindowLimiter(
            enabled=True, window_seconds=60, max_requests=1, clock=self.clock
        )
        self.assertTrue(limiter.hit("demo").allowed)

        self.clock.advance(10)
        self.assertEqual(limiter.hit("demo").retry_after, 50)

        self.clock.advance(49)
        self.assertEqual(limiter.hit("demo").retry_after, 1)

        self.clock.advance(2)
        self.assertTrue(limiter.hit("demo").allowed)

    def test_keys_do_not_share_budget(self):
        for _ in range(3):
            self.limiter.hit("demo")

        self.assertFalse(self.limiter.hit("demo").allowed)
        self.assertTrue(self.limiter.hit("alice").allowed)

    def test_rejected_request_does_not_extend_the_penalty(self):
        limiter = SlidingWindowLimiter(
            enabled=True, window_seconds=30, max_requests=1, clock=self.clock
        )
        self.assertTrue(limiter.hit("demo").allowed)

        # 一个死循环的客户端每 5 秒重试一次。被拒的请求不记账，
        # 否则它会把自己永久锁死，窗口永远滑不出去。
        for _ in range(4):
            self.clock.advance(5)
            self.assertFalse(limiter.hit("demo").allowed)

        self.clock.advance(10)
        self.assertTrue(limiter.hit("demo").allowed)

    def test_disabled_limiter_allows_everything(self):
        limiter = SlidingWindowLimiter(
            enabled=False, window_seconds=60, max_requests=1, clock=self.clock
        )
        for _ in range(50):
            self.assertTrue(limiter.hit("demo").allowed)

    def test_zero_max_requests_means_unlimited(self):
        limiter = SlidingWindowLimiter(
            enabled=True, window_seconds=60, max_requests=0, clock=self.clock
        )
        for _ in range(20):
            self.assertTrue(limiter.hit("demo").allowed)

    def test_reset_clears_one_key_or_all(self):
        for _ in range(3):
            self.limiter.hit("demo")
            self.limiter.hit("alice")

        self.limiter.reset("demo")
        self.assertTrue(self.limiter.hit("demo").allowed)
        self.assertFalse(self.limiter.hit("alice").allowed)

        self.limiter.reset()
        self.assertTrue(self.limiter.hit("alice").allowed)

    def test_sweep_drops_keys_that_left_the_window(self):
        from server.rate_limit import _MAX_TRACKED_KEYS

        limiter = SlidingWindowLimiter(
            enabled=True,
            window_seconds=60,
            max_requests=3,
            clock=self.clock,
        )
        for index in range(_MAX_TRACKED_KEYS + 1):
            limiter.hit(f"client-{index}")
        self.assertEqual(len(limiter._hits), _MAX_TRACKED_KEYS + 1)

        # 所有旧 key 都滑出窗口之后再打一发，触发清扫。
        self.clock.advance(600)
        limiter.hit("fresh")

        self.assertEqual(list(limiter._hits), ["fresh"])

    def test_config_is_read_from_auth_yml(self):
        limiter = SlidingWindowLimiter()

        self.assertTrue(limiter.enabled)
        self.assertEqual(limiter.window_seconds, 60.0)
        self.assertEqual(limiter.max_requests, 20)


if __name__ == "__main__":
    unittest.main()
