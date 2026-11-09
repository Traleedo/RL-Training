"""代码执行工具：模型参数 → 远端沙箱 → 归因返回。

职责边界（与 ``core.sandbox_client`` 的分工，见该模块 docstring）
----------------------------------------------------------------
归这里：参数校验（模型给的 JSON 永远不可信）、结果归因
（每种失败形态一句模型能定位的话）、metrics 记录。
归 client：URL 轮询、重试、并发、超时分层、响应规范化。

奖励约定：execute 层 reward 恒 0.0（outcome-only，奖励由轨迹末端
scorer 计算）；做 turn-level 惩罚（超时扣分等）时只改 execute 返回值，
``calc_reward`` 不用动。
"""

from __future__ import annotations

import asyncio

from core.base_tool import (
    BaseTool,
    OpenAIFunctionParametersSchema,
    OpenAIFunctionPropertySchema,
    OpenAIFunctionSchema,
    OpenAIFunctionToolSchema,
    ToolResult,
    ToolStatus,
)
from core.registry import register
from core.sandbox_client import RunResult, SandboxClient


def _attribute(run: RunResult, run_timeout: int) -> str:
    """把沙箱结果翻译成给模型看的文本。

    最重要的设计约束：每个失败形态必须**可归因**——模型要能区分
    "我代码错了"（stderr traceback）和"我跑太久了"（超时）和
    "没输出"（忘了 print）。全部折叠成一个 "error" 会让梯度变噪声。
    """
    if run.status is ToolStatus.TIMEOUT:
        return f"Error: execution timed out after {run_timeout}s"

    parts: list[str] = []
    if run.stdout.strip():
        parts.append(run.stdout.rstrip())
    if run.stderr.strip():
        parts.append("--- stderr ---\n" + run.stderr.lstrip())
    text = "\n".join(parts).strip()
    if not text:
        text = "(execution succeeded, no output — did you forget print()?)"
    return text


@register("tool", "python_exec")
class PythonExecTool(BaseTool):
    """执行 ``parameters["code"]`` 里的 Python 代码，返回 stdout/stderr。

    只吃结构化参数（function calling 风格）。围栏 ```` ```python ````
    提取属于 tool_parser（rollout 循环层）的职责——它负责把围栏内容
    填进 ``parameters["code"]``，两种风格因此共用同一个工具实现。
    """

    def __init__(self, config: dict):
        super().__init__("python_exec",
                         description="Execute Python code and return stdout/stderr")
        self._timeout = int(config.get("run_timeout", 10))
        self._client = SandboxClient(
            config["sandbox_url"],
            run_timeout=self._timeout,
            token=config.get("token") or None,
            max_retries=int(config.get("max_retries", 2)),
            backoff_base=float(config.get("backoff_base", 0.3)),
        )
        self._opened = False
        self._open_lock = asyncio.Lock()

    async def _ensure_open(self) -> None:
        # ClientSession 必须在 event loop 里创建；工具在配置阶段构造，
        # 那时可能还没有 loop，所以延迟到第一次 execute。进程级只开一次。
        async with self._open_lock:
            if not self._opened:
                await self._client.open()
                self._opened = True

    def get_openai_schema(self) -> OpenAIFunctionToolSchema:
        """发给模型的工具说明。description 是 prompt 的一部分，不是注释。"""
        return OpenAIFunctionToolSchema(
            type="function",
            function=OpenAIFunctionSchema(
                name="python_exec",
                description="Execute Python code and return stdout/stderr. "
                            "Print results with print().",
                parameters=OpenAIFunctionParametersSchema(
                    type="object",
                    properties={
                        "code": OpenAIFunctionPropertySchema(
                            type="string",
                            description="Complete Python code to run.",
                        ),
                    },
                    required=["code"],
                ),
            ),
        )

    async def execute(self, instance_id: str, parameters: dict, **kwargs) -> ToolResult:
        code = parameters.get("code")
        if not isinstance(code, str):
            # 不抛异常：错误文本回给模型自我修正，否则 gather 连坐整批轨迹
            return ToolResult(
                text="Error: missing or invalid required parameter 'code' (string)",
                status=ToolStatus.BAD_PARAMS,
                metrics={"bad_params": 1},
            )

        await self._ensure_open()
        run = await self._client.run_code(code)

        if not run.usable:
            # 基础设施失败：不进轨迹（上层丢弃），绝不变成 reward=0
            return ToolResult(text="", status=ToolStatus.INFRA, metrics=run.metrics)

        return ToolResult(
            text=_attribute(run, self._timeout),
            status=ToolStatus.OK,
            metrics={
                "return_code": float(run.return_code) if run.return_code is not None else -1.0,
                "timed_out": float(run.status is ToolStatus.TIMEOUT),
                "code_chars": float(len(code)),
                **run.metrics,
            },
        )
