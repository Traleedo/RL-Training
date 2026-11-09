"""``components.tools.python_exec`` 测试。

**不 mock SandboxClient** —— 要测的是工具的提取/归因/状态分类，
用 ``aiohttp.test_utils`` 起真实假沙箱（复用 test_sandbox_client 的
``FakeSandbox``），走完整 HTTP 链路。
"""

from __future__ import annotations

import asyncio

from aiohttp import web
from aiohttp.test_utils import TestServer

from components.tools.python_exec import PythonExecTool
from core.base_tool import ToolStatus
from tests.test_sandbox_client import FakeSandbox, _ok


def _timeout_envelope(seconds: float = 5.0) -> dict:
    return {
        "status": "Success",
        "message": "",
        "run_result": {
            "status": "TimeLimitExceeded",
            "execution_time": seconds,
            "return_code": None,
            "stdout": "",
            "stderr": "",
        },
    }


async def _tool_scenario(fn, fake: FakeSandbox, config: dict | None = None):
    """起一个假沙箱 → 构造 PythonExecTool 指向它 → 执行 fn(tool, fake) → 清理。"""
    app = web.Application()
    app.router.add_post("/run_code", fake.handle)
    server = TestServer(app)
    await server.start_server()
    tool = None
    try:
        cfg = {"sandbox_url": str(server.make_url("")).rstrip("/"),
               "run_timeout": 5, **(config or {})}
        tool = PythonExecTool(cfg)
        return await fn(tool, fake)
    finally:
        if tool is not None and tool._opened:
            await tool._client.close()
        await server.close()


def _run(fn, fake: FakeSandbox, config: dict | None = None):
    return asyncio.run(_tool_scenario(fn, fake, config))


# ---- 参数校验 -----------------------------------------------------------

def test_missing_code_param():
    """缺参数：BAD_PARAMS + 错误文本回给模型，绝不抛异常。"""
    fake = FakeSandbox()

    async def go(tool, fake):
        return await tool.execute("t0", {})

    result = _run(go, fake)
    assert result.status is ToolStatus.BAD_PARAMS
    assert "'code'" in result.text
    assert fake.requests == 0            # 根本不该发请求


def test_non_string_code_param():
    fake = FakeSandbox()

    async def go(tool, fake):
        return await tool.execute("t0", {"code": 123})

    result = _run(go, fake)
    assert result.status is ToolStatus.BAD_PARAMS
    assert fake.requests == 0


# ---- 正常路径与归因 -----------------------------------------------------

def test_ok_stdout_forwarded():
    """正常执行：stdout 原样返回，code 原样进请求体。"""
    fake = FakeSandbox(plan=[_ok(stdout="42\n", rc=0)])

    async def go(tool, fake):
        return await tool.execute("t0", {"code": "print(6*7)"})

    result = _run(go, fake)
    assert result.status is ToolStatus.OK
    assert result.text == "42"          # 尾部换行被 rstrip（print 的 \n 是噪声）
    assert result.reward == 0.0                        # outcome-only
    assert result.metrics["return_code"] == 0.0
    assert fake.bodies[0]["code"] == "print(6*7)"


def test_code_error_keeps_traceback():
    """模型代码自己报错：OK 状态 + stderr 可见（训练信号，不是 INFRA）。"""
    fake = FakeSandbox(plan=[_ok(
        stdout="", rc=1,
        stderr="Traceback ...\nZeroDivisionError: division by zero\n")])

    async def go(tool, fake):
        return await tool.execute("t0", {"code": "1/0"})

    result = _run(go, fake)
    assert result.status is ToolStatus.OK
    assert "ZeroDivisionError" in result.text
    assert "--- stderr ---" in result.text
    assert result.metrics["return_code"] == 1.0


def test_silent_success_gets_print_hint():
    """执行成功但没输出：给一句可归因的提示，别让模型面对空字符串发懵。"""
    fake = FakeSandbox(plan=[_ok(stdout="", rc=0)])

    async def go(tool, fake):
        return await tool.execute("t0", {"code": "x = 1 + 1"})

    result = _run(go, fake)
    assert result.status is ToolStatus.OK
    assert "did you forget print()" in result.text


def test_stdout_and_stderr_joined():
    fake = FakeSandbox(plan=[_ok(stdout="partial\n", rc=1, stderr="boom\n")])

    async def go(tool, fake):
        return await tool.execute("t0", {"code": "print('partial'); 1/0"})

    result = _run(go, fake)
    assert result.text.startswith("partial")
    assert "boom" in result.text


# ---- 超时与基础设施 -----------------------------------------------------

def test_timeout_is_attributable():
    """超时是给模型的信号（TIMEOUT），不是 INFRA。"""
    fake = FakeSandbox(plan=[_timeout_envelope()])

    async def go(tool, fake):
        return await tool.execute("t0", {"code": "while True: pass"})

    result = _run(go, fake)
    assert result.status is ToolStatus.OK          # TIMEOUT 是可归因结果，走 OK 通道
    assert "timed out after 5s" in result.text
    assert result.metrics["timed_out"] == 1.0


def test_infra_after_retries():
    """沙箱持续 503：INFRA、不可用（上层丢轨迹），重试发生在 client 层。"""
    fake = FakeSandbox(plan=[503, 503, 503])       # max_retries=2 → 共 3 次请求

    async def go(tool, fake):
        return await tool.execute("t0", {"code": "print(1)"})

    result = _run(go, fake, {"max_retries": 2, "backoff_base": 0.01})
    assert result.status is ToolStatus.INFRA
    assert not result.usable
    assert fake.requests == 3
