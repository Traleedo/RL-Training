"""Minimal HTTP code sandbox, sandbox-fusion compatible.

Mimics the sandbox-fusion ``/run_code`` response envelope so that tool code
written against sandbox-fusion (e.g. verl's tutorial SandboxTool) works
unchanged against this server.

Isolation level: process-level only (subprocess + rlimits + timeout).
Good enough to contain runaway code from your own model (infinite loops,
OOM, file spam). NOT a security boundary against a real adversary -- for
that use the sandbox-fusion Docker image on a separate machine.

Run:
    python scripts/mini_sandbox.py --port 8080

Test:
    curl http://127.0.0.1:8080/run_code -H "Content-Type: application/json" \\
        --data-raw '{"code": "print(1+1)", "language": "python"}'

Auth (optional): set SANDBOX_TOKEN env var; clients must send
    Authorization: Bearer <token>
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

MAX_TIMEOUT = 30          # hard cap on per-run wall time (seconds)
MAX_OUTPUT_CHARS = 4000   # truncate stdout/stderr before returning to model
MAX_CODE_CHARS = 20_000   # refuse absurdly large payloads

app = FastAPI(title="mini-sandbox")


class RunRequest(BaseModel):
    code: str = Field(..., description="source code to execute")
    language: str = Field("python", description="only 'python' supported")
    run_timeout: int = Field(10, ge=1, le=MAX_TIMEOUT)
    stdin: str = ""


class RunResult(BaseModel):
    status: str
    execution_time: float
    return_code: int | None
    stdout: str
    stderr: str


class RunResponse(BaseModel):
    status: str
    message: str
    run_result: RunResult | None


def _apply_rlimits() -> None:  # pragma: no cover - runs in child process
    """Linux-only limits applied inside the child before exec."""
    import resource  # noqa: PLC0415  -- unix-only module, import lazily

    try:
        # 512 MB address space cap; generous for python startup (~80MB)
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024,) * 2)
        # 20s CPU time cap, independent of wall-time kill
        resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
        # no more than 64 subprocesses / threads
        resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
        # max 256 MB output file size, avoids filling the disk
        resource.setrlimit(resource.RLIMIT_FSIZE, (256 * 1024 * 1024,) * 2)
    except (ValueError, OSError):
        pass  # some rlimits may not be raisable in the current context


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    half = MAX_OUTPUT_CHARS // 2
    return (
        text[:half]
        + f"\n... [truncated {len(text) - MAX_OUTPUT_CHARS} chars] ...\n"
        + text[-half:]
    )


async def _execute(code: str, run_timeout: int, stdin: str) -> RunResult:
    workdir = Path(tempfile.gettempdir()) / f"sandbox-{uuid.uuid4().hex}"
    workdir.mkdir(parents=True, exist_ok=True)
    script = workdir / "main.py"
    script.write_text(code, encoding="utf-8")

    start = time.monotonic()
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-I", str(script),
            cwd=str(workdir),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            preexec_fn=_apply_rlimits if os.name == "posix" else None,
            env={"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8",
                 "HOME": str(workdir), "TMPDIR": str(workdir)},
        )
        try:
            out, err = await asyncio.wait_for(
                proc.communicate(stdin.encode("utf-8")), timeout=run_timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return RunResult(status="TimeLimitExceeded", execution_time=run_timeout,
                             return_code=None, stdout="", stderr=f"timed out after {run_timeout}s")
        elapsed = time.monotonic() - start
        status = "Finished" if proc.returncode == 0 else "RuntimeError"
        # don't leak host paths in tracebacks shown to the model
        err_text = err.decode("utf-8", "replace").replace(str(workdir), "sandbox")
        out_text = out.decode("utf-8", "replace").replace(str(workdir), "sandbox")
        return RunResult(status=status, execution_time=round(elapsed, 4),
                         return_code=proc.returncode,
                         stdout=_truncate(out_text),
                         stderr=_truncate(err_text))
    finally:
        # best-effort cleanup of the per-run scratch dir
        import shutil
        shutil.rmtree(workdir, ignore_errors=True)


@app.post("/run_code", response_model=RunResponse)
async def run_code(req: RunRequest, authorization: str = Header("")):
    _check_auth(authorization)
    if req.language not in ("python", "python3"):
        raise HTTPException(400, f"language '{req.language}' not supported, only python")
    if len(req.code) > MAX_CODE_CHARS:
        raise HTTPException(413, f"code too long (> {MAX_CODE_CHARS} chars)")
    result = await _execute(req.code, req.run_timeout, req.stdin)
    return RunResponse(status="Success", message="", run_result=result)


@app.get("/health")
async def health():
    return {"status": "ok"}


def _check_auth(authorization: str) -> None:
    token = os.environ.get("SANDBOX_TOKEN", "")
    if not token:
        return
    if authorization != f"Bearer {token}":
        raise HTTPException(401, "invalid token")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind 0.0.0.0 only behind a firewall/tunnel")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
