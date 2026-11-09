"""沙箱 HTTP 客户端 —— 工具与远端执行环境之间唯一的传输层。

职责边界（与 ``core.base_tool`` 的分工）
----------------------------------------
归这里：URL 轮询、连接池、重试、并发上限、超时分层、
HTTP 状态码 → ``ToolStatus`` 映射、响应规范化（不同后端 → 同一 ``RunResult``）、
输出截断与路径脱敏。

归工具（``BaseTool.execute``）：从模型文本里提取代码、参数校验、
错误文本组装、超时是否计入奖励。

协议
----
与 sandbox-fusion 的 ``POST /run_code`` 对齐（多出的字段对方会忽略），
**不在这个基础上发明字段** —— 保持一个后端方言，换后端时只写适配器。

请求::

    POST {base_url}/run_code
    Content-Type: application/json
    Authorization: Bearer <token>          # 服务端开启鉴权时才需要

    {"code": "...", "language": "python", "run_timeout": 10, "stdin": ""}

响应::

    {"status": "Success", "message": "",
     "run_result": {"status": "Finished", "execution_time": 0.14,
                    "return_code": 0, "stdout": "...", "stderr": ""}}

超时分层（四层，必须逐层变宽）
------------------------------
    单次 HTTP 连接超时   connect=2s              ← 这里
    单次 HTTP 总超时     run_timeout + 5s        ← 这里
    沙箱内执行超时       run_timeout             ← 请求体里传给沙箱
    轨迹级工具预算       max_tool_seconds        ← tool_agent 循环（P2）

原则：外层永远比内层宽，且外层失败要能归因到内层。反了就会出现
「沙箱还在跑，客户端已经超时」，结果是重复执行 + 假 INFRA。
"""

from __future__ import annotations

import asyncio
import itertools
import random
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from core.base_tool import ToolStatus

__all__ = ["SandboxClient", "SandboxAuthError", "RunResult"]


class SandboxAuthError(RuntimeError):
    """沙箱鉴权失败（401/403）。

    这是**配置错误**，不是运行时故障 —— 必须在启动阶段炸掉，
    而不是训练 200 步之后才发现奖励全是零。
    """


@dataclass(frozen=True)
class RunResult:
    """规范化后的单次代码执行结果。"""

    status: ToolStatus          # 已映射，不是后端原始字符串
    stdout: str = ""
    stderr: str = ""
    return_code: int | None = None
    elapsed: float = 0.0        # 沙箱自报的执行时间（秒）
    raw_status: str = ""        # 后端原始 status，仅用于 metrics 与日志
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        """与 ``ToolResult.usable`` 同义：INFRA 时调用方应丢弃该轨迹。"""
        return self.status is not ToolStatus.INFRA


# 宿主机路径脱敏：匹配 ``.../sandbox-<hex>/...`` 形态的绝对路径（含 Windows 盘符）。
# 沙箱 traceback 会带宿主机临时目录路径，不该让模型看到。
_SANDBOX_PATH_RE = re.compile(
    r"[^\s\"'`]*sandbox-[0-9a-fA-F]{8,}[^\s\"'`]*"
)


def _sanitize(text: str) -> str:
    return _SANDBOX_PATH_RE.sub("sandbox/main.py", text)


def _truncate(text: str, limit: int) -> str:
    """头尾各留一半，中间标注省略量 —— 报错常在开头，结果常在结尾。"""
    if len(text) <= limit:
        return text
    half = max(limit // 2, 1)
    return (
        text[:half]
        + f"\n... [truncated {len(text) - limit} chars] ...\n"
        + text[-half:]
    )


# ---------------------------------------------------------------------
# HTTP 状态码 → 处理策略
# ---------------------------------------------------------------------
#: 可重试：网络层故障或明确的过载信号
_RETRYABLE_HTTP = frozenset({429, 502, 503, 504})
#: 请求写错了，重试没有意义
_BAD_REQUEST_HTTP = frozenset({400, 404, 405, 413, 422})


class SandboxClient:
    """远端代码沙箱的异步客户端。

    用法::

        async with SandboxClient(["http://127.0.0.1:8080"]) as client:
            result = await client.run_code("print(1+1)")

    ``SandboxClient`` 是**进程级**资源：session 在 ``open()``（即
    ``__aenter__``）里建 —— ``ClientSession`` 必须在运行中的 event loop 里
    创建，而组件构造阶段通常还没有 loop。不要放在 ``__init__``。
    工具类持有 client 注入的 client，自己的 ``create/release`` 留给真正
    per-trajectory 的东西（有状态工具的容器）。
    """

    def __init__(
        self,
        base_urls: str | list[str],
        *,
        run_timeout: int = 10,
        max_concurrency: int = 32,
        max_retries: int = 2,
        backoff_base: float = 0.3,
        token: str | None = None,
        max_output_chars: int = 4000,
    ) -> None:
        self._urls = [base_urls] if isinstance(base_urls, str) else list(base_urls)
        if not self._urls:
            raise ValueError("base_urls 不能为空")
        self._run_timeout = int(run_timeout)
        self._max_concurrency = int(max_concurrency)
        self._max_retries = int(max_retries)
        self._backoff_base = float(backoff_base)
        self._token = token
        self._max_output_chars = int(max_output_chars)

        self._session: Any = None          # aiohttp.ClientSession（惰性导入）
        self._rr = itertools.count()       # round-robin 计数器
        self._sem: Any = None              # asyncio.Semaphore
        self._stats: dict[str, float] = {}
        self._elapsed: deque[float] = deque(maxlen=1024)
        self._url_counts: dict[str, int] = {}

    # ------------------------------------------------------------------
    # 生命周期（进程级）
    # ------------------------------------------------------------------
    async def open(self) -> None:
        import aiohttp  # 惰性导入：不用沙箱时不需要安装

        # 两道闸门的分工：Semaphore 限「排队」（超了在本地等，不让请求打出去），
        # TCPConnector.limit 限「连接」（防半开连接堆积）。只做后者会看到大量
        # 「Connection pool is full」警告，只做前者则长连接数不受控。
        total = self._max_concurrency * len(self._urls)
        connector = aiohttp.TCPConnector(
            limit=total, limit_per_host=self._max_concurrency
        )
        headers = {}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        self._session = aiohttp.ClientSession(
            connector=connector,
            headers=headers,
            timeout=aiohttp.ClientTimeout(
                total=self._run_timeout + 5,
                connect=2,
                sock_read=self._run_timeout + 4,
            ),
        )
        self._sem = asyncio.Semaphore(total)

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def __aenter__(self) -> "SandboxClient":
        await self.open()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # 执行
    # ------------------------------------------------------------------
    async def run_code(
        self,
        code: str,
        *,
        language: str = "python",
        stdin: str = "",
    ) -> RunResult:
        """执行一段代码。只重试幂等请求 —— 无状态沙箱的 ``/run_code`` 是幂等的；
        一旦沙箱变成有状态（文件延续、Jupyter 会话），重试会重复副作用，
        ``_is_retryable`` 那张表要整体重审。
        """
        if self._session is None:
            raise RuntimeError(
                "SandboxClient 未打开：请用 async with 或先 await client.open()"
            )
        if not isinstance(code, str) or not code.strip():
            # 编程错误（不是模型错误），直接抛 —— 该由工具侧在调用前拦
            raise ValueError("code 必须是非空字符串")

        payload = {
            "code": code,
            "language": language,
            "run_timeout": self._run_timeout,
            "stdin": stdin,
        }
        retries = 0
        for attempt in range(self._max_retries + 1):
            url = self._urls[next(self._rr) % len(self._urls)]
            self._stats["requests"] = self._stats.get("requests", 0) + 1
            self._url_counts[url] = self._url_counts.get(url, 0) + 1
            try:
                async with self._sem:
                    async with self._session.post(
                        f"{url}/run_code", json=payload
                    ) as resp:
                        if resp.status in (401, 403):
                            raise SandboxAuthError(
                                f"沙箱鉴权失败（HTTP {resp.status}，{url}）。"
                                f"这是配置错误：检查 token 是否与 "
                                f"SANDBOX_TOKEN 环境变量一致。"
                            )
                        if resp.status == 429:
                            retriable = True
                        elif resp.status in _RETRYABLE_HTTP:
                            retriable = True
                        elif resp.status in _BAD_REQUEST_HTTP:
                            # 请求写错了，重试无意义
                            body = await resp.text()
                            return self._infra(
                                f"HTTP {resp.status}", url,
                                metrics={"http_status": resp.status},
                                stderr=body[:500],
                            )
                        elif resp.status >= 500:
                            # 500 是服务端自身 bug，重试无意义
                            retriable = False
                        elif resp.status != 200:
                            body = await resp.text()
                            return self._infra(
                                f"HTTP {resp.status}", url,
                                metrics={"http_status": resp.status},
                                stderr=body[:500],
                            )
                        else:
                            data = await resp.json(content_type=None)
                            return self._normalize(data, url, retries=retries)

                if not retriable:
                    return self._infra(f"HTTP {resp.status}", url,
                                       metrics={"http_status": resp.status})
                # 走到这里说明是可重试错误且还有余量
                if attempt < self._max_retries:
                    retries += 1
                    self._stats["retries"] = self._stats.get("retries", 0) + 1
                    await self._backoff(attempt)
                    continue
                return self._infra(f"HTTP {resp.status}", url,
                                   metrics={"http_status": resp.status,
                                            "retries": retries})
            except SandboxAuthError:
                raise
            except (asyncio.TimeoutError, TimeoutError):
                if attempt < self._max_retries:
                    retries += 1
                    self._stats["retries"] = self._stats.get("retries", 0) + 1
                    await self._backoff(attempt)
                    continue
                return self._infra(
                    f"timeout after {self._run_timeout + 5}s", url,
                    metrics={"retries": retries},
                )
            except Exception as exc:  # 连接拒绝 / DNS 失败 / 协议错误
                if self._is_conn_error(exc) and attempt < self._max_retries:
                    retries += 1
                    self._stats["retries"] = self._stats.get("retries", 0) + 1
                    await self._backoff(attempt)
                    continue
                if self._is_conn_error(exc):
                    return self._infra(
                        f"{type(exc).__name__}: {exc}"[:300], url,
                        metrics={"retries": retries},
                    )
                raise  # 非网络类异常（编程错误）不吞

        return self._infra("exhausted", self._urls[0], metrics={"retries": retries})

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    async def _backoff(self, attempt: int) -> None:
        await asyncio.sleep(
            self._backoff_base * (2**attempt) + random.uniform(0, 0.05)
        )

    @staticmethod
    def _is_conn_error(exc: Exception) -> bool:
        mod = type(exc).__module__
        return mod.startswith("aiohttp") or isinstance(
            exc, (ConnectionError, OSError)
        )

    def _infra(
        self, message: str, url: str, *,
        metrics: dict[str, float] | None = None,
        stderr: str = "",
    ) -> RunResult:
        self._stats[f"status_{ToolStatus.INFRA.value}"] = (
            self._stats.get(f"status_{ToolStatus.INFRA.value}", 0) + 1
        )
        m = dict(metrics or {})
        m["infra"] = 1
        return RunResult(
            status=ToolStatus.INFRA,
            stderr=stderr or f"sandbox infra failure ({message}) at {url}",
            raw_status=message,
            metrics=m,
        )

    def _normalize(
        self, data: dict[str, Any], url: str, *, retries: int = 0
    ) -> RunResult:
        """按 sandbox-fusion envelope 解响应；解不动一律 INFRA。"""
        outer = data.get("status")
        if outer != "Success":
            return self._infra(
                f"outer status={outer!r}", url,
                stderr=str(data.get("message", ""))[:500],
            )
        rr = data.get("run_result")
        if not isinstance(rr, dict):
            return self._infra("missing run_result", url)

        raw_status = str(rr.get("status", ""))
        elapsed = float(rr.get("execution_time", 0.0) or 0.0)
        return_code = rr.get("return_code")
        stdout = _sanitize(str(rr.get("stdout", "") or ""))
        stderr = _sanitize(str(rr.get("stderr", "") or ""))

        if raw_status in ("Finished", "RuntimeError"):
            # 两者都是 OK：代码自己报错（RuntimeError, rc≠0）是**训练信号**，
            # 不是基础设施故障 —— stderr 照常回给模型自我修正。
            status = ToolStatus.OK
        elif raw_status == "TimeLimitExceeded":
            status = ToolStatus.TIMEOUT
        else:
            # 未知状态才 INFRA；遇到新后端方言时在这里补映射
            return self._infra(f"run_result.status={raw_status!r}", url)

        self._elapsed.append(elapsed)
        self._stats[f"status_{status.value}"] = (
            self._stats.get(f"status_{status.value}", 0) + 1
        )
        return RunResult(
            status=status,
            stdout=_truncate(stdout, self._max_output_chars),
            stderr=_truncate(stderr, self._max_output_chars),
            return_code=int(return_code) if return_code is not None else None,
            elapsed=elapsed,
            raw_status=raw_status,
            metrics={"retries": retries} if retries else {},
        )

    # ------------------------------------------------------------------
    # 度量
    # ------------------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        """累计统计。p95 是决定要不要上异步调度的那根线（<1s 共址同步够用）。"""
        out: dict[str, Any] = dict(self._stats)
        if self._elapsed:
            xs = sorted(self._elapsed)
            n = len(xs)
            out["elapsed_p50"] = xs[n // 2]
            out["elapsed_p95"] = xs[min(n - 1, int(n * 0.95))]
        out["url_counts"] = dict(self._url_counts)
        return out
