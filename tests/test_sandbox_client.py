"""``core.sandbox_client`` POST 逻辑测试。

**不 mock client 本身** —— 要测的正是 HTTP 层行为（状态码、重试、超时、并发），
mock 掉就什么都没验证。用 ``aiohttp.test_utils`` 起一个真实的本地假后端。

注意：TestServer 绑定在它 start 的那个 event loop 上，loop 关了 socket 就没了。
所以每个用例把「起 server → 开 client → 执行 → 关」包在**同一次**
``asyncio.run`` 里（见 ``_scenario``）。
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from core.base_tool import ToolStatus
from core.sandbox_client import RunResult, SandboxAuthError, SandboxClient


def _ok(stdout: str = "hello\n", rc: int = 0, elapsed: float = 0.01,
        stderr: str = "") -> dict:
    return {
        "status": "Success",
        "message": "",
        "run_result": {
            "status": "Finished",
            "execution_time": elapsed,
            "return_code": rc,
            "stdout": stdout,
            "stderr": stderr,
        },
    }


class FakeSandbox:
    """可编程假后端。``plan`` 是响应序列：int → HTTP 状态码；dict → JSON envelope。

    ``plan`` 耗尽后一律返回成功 envelope。``track_concurrency=True`` 时记录
    服务端观察到的最大同时在飞请求数。
    """

    def __init__(self, plan=None, *, track_concurrency: bool = False) -> None:
        self.plan = list(plan or [])
        self.requests = 0
        self.bodies: list[dict] = []
        self.track_concurrency = track_concurrency
        self.in_flight = 0
        self.max_in_flight = 0
        self.server: TestServer | None = None

    async def handle(self, request: web.Request) -> web.Response:
        self.requests += 1
        self.bodies.append(await request.json())
        if self.track_concurrency:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            try:
                await asyncio.sleep(0.05)
            finally:
                self.in_flight -= 1
        step = self.requests - 1
        if step < len(self.plan):
            item = self.plan[step]
            if isinstance(item, int):
                return web.json_response({"error": "planned"}, status=item)
            return web.json_response(item)
        return web.json_response(_ok())


async def _scenario(
    fn: Callable[..., Awaitable],
    fakes: list[FakeSandbox],
    client_kwargs: dict | None = None,
):
    """起 N 个假 server → 开一个 client 指向它们 → 执行 fn(client, urls) → 全部关掉。"""
    servers: list[TestServer] = []
    client: SandboxClient | None = None
    try:
        urls = []
        for fake in fakes:
            app = web.Application()
            app.router.add_post("/run_code", fake.handle)
            server = TestServer(app)
            await server.start_server()
            servers.append(server)
            urls.append(str(server.make_url("")).rstrip("/"))
        client = SandboxClient(urls[0] if len(urls) == 1 else urls,
                               **(client_kwargs or {}))
        await client.open()
        return await fn(client, urls)
    finally:
        if client is not None:
            await client.close()
        for server in servers:
            await server.close()


# ---- 正常路径 -----------------------------------------------------------

def test_ok_result():
    fake = FakeSandbox()

    async def go(client, urls):
        return await client.run_code("print(1+1)")

    result = asyncio.run(_scenario(go, [fake]))
    assert isinstance(result, RunResult)
    assert result.status is ToolStatus.OK
    assert result.stdout == "hello\n"
    assert result.return_code == 0
    assert fake.requests == 1
    assert fake.bodies[0]["code"] == "print(1+1)"
    assert fake.bodies[0]["run_timeout"] == 10     # 默认超时进 payload


def test_code_error_is_ok_status_with_return_code():
    """代码自己报错是**正常返回**：status=OK、return_code≠0、stderr 带 traceback。

    它是训练信号，不是基础设施故障 —— 绝不能归类成 INFRA。
    """
    fake = FakeSandbox(plan=[_ok(
        stderr="Traceback ...\nZeroDivisionError: division by zero\n", rc=1)])

    async def go(client, urls):
        return await client.run_code("1/0")

    result = asyncio.run(_scenario(go, [fake]))
    assert result.status is ToolStatus.OK
    assert result.return_code == 1
    assert "ZeroDivisionError" in result.stderr
    assert fake.requests == 1                      # 不重试


def test_sandbox_timeout_is_timeout_not_infra():
    """沙箱自报 TimeLimitExceeded 是合法执行结果：不重试、不是 INFRA。"""
    fake = FakeSandbox(plan=[{
        "status": "Success", "message": "",
        "run_result": {"status": "TimeLimitExceeded", "execution_time": 10.0,
                       "return_code": None, "stdout": "", "stderr": "timed out"},
    }])

    async def go(client, urls):
        return await client.run_code("while True: pass")

    result = asyncio.run(_scenario(go, [fake]))
    assert result.status is ToolStatus.TIMEOUT
    assert fake.requests == 1


# ---- 重试表 -------------------------------------------------------------

def test_conn_refused_retries_then_infra():
    """连接失败：重试耗尽后 INFRA，metrics 记录重试次数。"""
    # 这个用例不起 server：直接把 client 指向一个没有进程监听的端口
    async def main():
        client = SandboxClient(
            "http://127.0.0.1:59907", max_retries=1, backoff_base=0.01
        )
        try:
            await client.open()
            return await client.run_code("print(1)")
        finally:
            await client.close()

    result = asyncio.run(main())
    assert result.status is ToolStatus.INFRA
    assert result.metrics.get("retries") == 1
    assert result.usable is False


def test_503_then_success_counts_retry():
    fake = FakeSandbox(plan=[503])

    async def go(client, urls):
        return await client.run_code("print('ok')")

    result = asyncio.run(_scenario(go, [fake]))
    assert result.status is ToolStatus.OK
    assert fake.requests == 2
    assert result.metrics.get("retries") == 1


def test_500_not_retried():
    """500 是服务端自身 bug，重试无意义。"""
    fake = FakeSandbox(plan=[500])

    async def go(client, urls):
        return await client.run_code("print('x')")

    result = asyncio.run(_scenario(go, [fake]))
    assert result.status is ToolStatus.INFRA
    assert fake.requests == 1


def test_401_raises_config_error_immediately():
    """鉴权失败是配置错误：立刻抛、绝不静默重试 —— 否则训练 200 步后
    才发现奖励全是零。"""
    fake = FakeSandbox(plan=[401, 401, 401])

    async def go(client, urls):
        return await client.run_code("print('x')")

    with pytest.raises(SandboxAuthError, match="SANDBOX_TOKEN"):
        asyncio.run(_scenario(go, [fake]))
    assert fake.requests == 1                      # 不重试


def test_bad_request_400_not_retried():
    fake = FakeSandbox(plan=[400])

    async def go(client, urls):
        return await client.run_code("print('x')")

    result = asyncio.run(_scenario(go, [fake]))
    assert result.status is ToolStatus.INFRA
    assert fake.requests == 1


def test_outer_status_not_success_is_infra():
    fake = FakeSandbox(plan=[{"status": "InternalError", "message": "boom"}])

    async def go(client, urls):
        return await client.run_code("print('x')")

    result = asyncio.run(_scenario(go, [fake]))
    assert result.status is ToolStatus.INFRA
    assert "boom" in result.stderr


# ---- 截断与脱敏 -----------------------------------------------------------

def test_output_truncation_marks_middle():
    fake = FakeSandbox(plan=[_ok(stdout="x" * 5000)])

    async def go(client, urls):
        return await client.run_code("print('big')")

    result = asyncio.run(_scenario(
        go, [fake], client_kwargs={"max_output_chars": 100}))
    assert len(result.stdout) < 200
    assert "truncated" in result.stdout
    assert result.stdout.startswith("x")           # 开头保留（报错常在开头）
    assert result.stdout.endswith("x")             # 结尾保留（结果常在结尾）


def test_host_path_sanitized_in_stderr():
    fake = FakeSandbox(plan=[_ok(
        stderr='File "C:\\Users\\1\\AppData\\Local\\Temp\\'
               'sandbox-81f5a0e3fcad417d853889db3f29c656\\main.py", line 1',
        rc=1,
    )])

    async def go(client, urls):
        return await client.run_code("1/0")

    result = asyncio.run(_scenario(go, [fake]))
    assert "sandbox/main.py" in result.stderr
    assert "C:" not in result.stderr               # 宿主机路径不外泄


# ---- 并发与多副本 -----------------------------------------------------------

def test_concurrency_capped_by_semaphore():
    """max_concurrency=2 时，20 个并发调用在服务端看到的在飞数 ≤ 2。"""
    fake = FakeSandbox(track_concurrency=True)

    async def go(client, urls):
        return await asyncio.gather(*[
            client.run_code(f"print({i})") for i in range(20)
        ])

    results = asyncio.run(_scenario(
        go, [fake], client_kwargs={"max_concurrency": 2, "max_retries": 0}))
    assert all(r.status is ToolStatus.OK for r in results)
    assert fake.max_in_flight <= 2


def test_round_robin_across_urls():
    """两个副本时请求轮流分发（池子水平扩展的前提）。"""
    fake_a, fake_b = FakeSandbox(), FakeSandbox()

    async def go(client, urls):
        await asyncio.gather(*[client.run_code(f"print({i})") for i in range(6)])

    asyncio.run(_scenario(go, [fake_a, fake_b]))
    assert fake_a.requests == 3
    assert fake_b.requests == 3


# ---- 校验 -----------------------------------------------------------------

def test_empty_code_raises_valueerror():
    """空 code 是编程错误（工具侧该拦），直接抛而不是发请求。"""
    fake = FakeSandbox()

    async def go(client, urls):
        try:
            await client.run_code("   ")
        except ValueError:
            return True
        return False

    assert asyncio.run(_scenario(go, [fake])) is True
    assert fake.requests == 0


def test_run_without_open_raises():
    client = SandboxClient("http://127.0.0.1:59908")
    with pytest.raises(RuntimeError, match="open"):
        asyncio.run(client.run_code("print(1)"))


def test_stats_report_counts_and_percentiles():
    fake = FakeSandbox(plan=[503])                 # 走一次重试路径

    async def go(client, urls):
        await client.run_code("print('s')")
        return client.stats()

    stats = asyncio.run(_scenario(go, [fake]))
    assert stats["requests"] == 2
    assert stats["retries"] == 1
    assert stats["status_ok"] == 1
    assert "elapsed_p95" in stats


# ---- 真沙箱方言：RuntimeError 是训练信号 ------------------------------------

def test_runtime_error_status_is_ok_not_infra():
    """真沙箱对 1/0 返回 run_result.status='RuntimeError', rc=1。

    这是模型的训练信号，绝不能归成 INFRA（曾因此把所有异常代码静默丢弃）。
    """
    fake = FakeSandbox(plan=[_ok(stderr="Traceback ...\nZeroDivisionError\n", rc=1)])
    # 模拟 mini_sandbox 真实行为：异常时 status 是 RuntimeError 而非 Finished
    fake.plan[0]["run_result"]["status"] = "RuntimeError"

    async def go(client, urls):
        return await client.run_code("1/0")

    result = asyncio.run(_scenario(go, [fake]))
    assert result.status is ToolStatus.OK
    assert result.return_code == 1
    assert "ZeroDivisionError" in result.stderr
