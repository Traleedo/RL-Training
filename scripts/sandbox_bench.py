"""沙箱效果压测 —— 对 mini_sandbox 与 sandbox-fusion 通用。

协议都走 sandbox-fusion 的 ``POST /run_code`` envelope（两者对齐），
输出分四段：smoke（功能）/ latency（延迟）/ concurrency（并发）/
probes（隔离行为）。全部结果带判定，不用人工翻 JSON。

云机用法::

    pip install aiohttp
    python scripts/sandbox_bench.py --url http://127.0.0.1:8080 --mode all

    # 分段跑：
    python scripts/sandbox_bench.py --url http://127.0.0.1:8080 --mode smoke
    python scripts/sandbox_bench.py --url http://127.0.0.1:8080 --mode latency --n 50
    python scripts/sandbox_bench.py --url http://127.0.0.1:8080 --mode concurrency --c 32
    python scripts/sandbox_bench.py --url http://127.0.0.1:8080 --mode probes
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time

try:
    import aiohttp
except ImportError:
    raise SystemExit("缺少 aiohttp：pip install aiohttp")

PASS, FAIL, WARN = "[PASS]", "[FAIL]", "[WARN]"


def _json(r) -> dict:
    """容错解析：隧道/网关返回 text/plain 502 时不炸，降级为可读 dict。"""
    try:
        return r.json(content_type=None)
    except Exception:
        return {"status": "BAD_RESPONSE", "run_result": {"stderr": f"HTTP {r.status}: 非JSON响应"}}


def _parse(data: dict) -> tuple[str, str, float]:
    """统一解析 envelope → (status, stdout+stderr, elapsed)。"""
    if not isinstance(data, dict):
        return "BAD_RESPONSE", str(data)[:200], 0.0
    rr = data.get("run_result") or {}
    status = str(rr.get("status") or data.get("status") or "Unknown")
    text = str(rr.get("stdout", "") or "") + str(rr.get("stderr", "") or "")
    return status, text, float(rr.get("execution_time", 0.0) or 0.0)


def _pctl(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(len(s) * q))]


# =====================================================================
# smoke：功能正确性
# =====================================================================
async def smoke(session: aiohttp.ClientSession, url: str) -> None:
    print("\n== smoke：功能正确性 ==")

    # 1. 基本执行
    t0 = time.monotonic()
    async with session.post(f"{url}/run_code",
                            json={"code": "print(1+1)", "language": "python",
                                  "run_timeout": 10}) as r:
        status, text, _ = _parse(await _json(r))
    wall = time.monotonic() - t0
    ok = "2" in text
    print(f"{PASS if ok else FAIL} 基本执行 print(1+1) → {text.strip()!r} "
          f"（wall {wall:.2f}s，沙箱自报 {status}）")

    # 2. 代码异常返回 stderr
    async with session.post(f"{url}/run_code",
                            json={"code": "1/0", "language": "python",
                                  "run_timeout": 10}) as r:
        status, text, _ = _parse(await _json(r))
    ok = "ZeroDivisionError" in text
    print(f"{PASS if ok else FAIL} 代码异常捕获 → {text.strip().splitlines()[-1][:60]!r}")

    # 3. import 可用性（沙箱环境的解释器里有没有常用包）
    for mod in ("numpy", "pandas", "requests", "sympy"):
        async with session.post(f"{url}/run_code",
                                json={"code": f"import {mod}",
                                      "language": "python", "run_timeout": 15}) as r:
            status, text, _ = _parse(await _json(r))
        available = status == "Finished"
        print(f"{WARN if available else '[-]'} import {mod}: "
              f"{'可用' if available else '不可用（需要时在沙箱环境里装）'}")

    # 4. 大输出截断（服务端应有上限，否则模型上下文会被打爆）
    async with session.post(f"{url}/run_code",
                            json={"code": "print('x' * 500000)",
                                  "language": "python", "run_timeout": 10}) as r:
        status, text, _ = _parse(await _json(r))
    ok = len(text) < 100_000
    print(f"{PASS if ok else WARN} 50 万字符输出 → 返回 {len(text)} 字符"
          f"{'（有截断）' if ok else '（未截断！接入模型前必须在工具侧截断）'}")


# =====================================================================
# latency：串行延迟分布
# =====================================================================
async def latency(session: aiohttp.ClientSession, url: str, n: int) -> None:
    print(f"\n== latency：串行 {n} 次 print ==")
    walls: list[float] = []
    execs: list[float] = []
    for _ in range(n):
        t0 = time.monotonic()
        async with session.post(f"{url}/run_code",
                                json={"code": "print(1)", "language": "python",
                                      "run_timeout": 10}) as r:
            status, _, ex = _parse(await _json(r))
        walls.append(time.monotonic() - t0)
        execs.append(ex)
    print(f"  wall   p50={_pctl(walls, 0.5)*1000:6.1f}ms  "
          f"p95={_pctl(walls, 0.95)*1000:6.1f}ms  "
          f"max={max(walls)*1000:6.1f}ms")
    if any(execs):
        print(f"  沙箱内 p50={_pctl(execs, 0.5)*1000:6.1f}ms  "
              f"p95={_pctl(execs, 0.95)*1000:6.1f}ms")
        print(f"  → 差值 = HTTP + 进程冷启动开销，越小说明沙箱越轻")
    verdict = "可同步共址" if _pctl(walls, 0.95) < 1.0 else "考虑分池/异步"
    print(f"  判定：p95 {_pctl(walls, 0.95)*1000:.0f}ms → {verdict}")


# =====================================================================
# concurrency：并发吞吐与正确率
# =====================================================================
async def concurrency(session: aiohttp.ClientSession, url: str,
                      c: int, rounds: int) -> None:
    print(f"\n== concurrency：{c} 并发 × {rounds} 轮 ==")
    codes = [f"print({i} * 7)" for i in range(c)]
    ok_count = 0
    fail_examples: list[str] = []
    t0 = time.monotonic()
    for _ in range(rounds):
        async def one(code: str) -> bool:
            try:
                async with session.post(f"{url}/run_code",
                                        json={"code": code, "language": "python",
                                              "run_timeout": 10}) as r:
                    status, text, _ = _parse(await _json(r))
                    return "Finished" in status
            except Exception as exc:
                fail_examples.append(f"{type(exc).__name__}: {exc}")
                return False
        results = await asyncio.gather(*[one(c) for c in codes])
        ok_count += sum(results)
    wall = time.monotonic() - t0
    total = c * rounds
    print(f"  成功 {ok_count}/{total}，总耗时 {wall:.1f}s，"
          f"吞吐 {total / wall:.1f} req/s")
    if fail_examples:
        print(f"  失败示例：{fail_examples[0][:120]}")
    if ok_count < total:
        print(f"  {WARN} 有失败 —— 单副本并发上限低于 {c}，"
              f"多卡场景需要加副本或客户端限流")
    else:
        print(f"  判定：单副本扛得住 {c} 并发")


# =====================================================================
# probes：隔离行为（预期行为对 mini_sandbox / sandbox-fusion 有差异，注明）
# =====================================================================
async def probes(session: aiohttp.ClientSession, url: str) -> None:
    print("\n== probes：隔离行为 ==")

    # 1. 死循环 → 必须被超时杀掉
    t0 = time.monotonic()
    async with session.post(f"{url}/run_code",
                            json={"code": "while True: pass",
                                  "language": "python", "run_timeout": 3}) as r:
        status, text, _ = _parse(await _json(r))
    wall = time.monotonic() - t0
    killed = "TimeLimit" in status or "timed out" in text.lower()
    print(f"{PASS if killed else FAIL} 死循环 3s 超时 → {status}，"
          f"wall {wall:.1f}s{'（按时杀掉）' if wall < 8 else '（杀慢了！）'}")

    # 2. 内存打爆 → 应被限制（rlimit 或容器 OOM），不能拖垮宿主机
    t0 = time.monotonic()
    async with session.post(f"{url}/run_code",
                            json={"code": "a = bytearray(8 * 1024**3)",
                                  "language": "python", "run_timeout": 10}) as r:
        status, text, _ = _parse(await _json(r))
    wall = time.monotonic() - t0
    contained = status != "Finished" or "MemoryError" in text
    print(f"{PASS if contained else FAIL} 申请 8GB 内存 → {status}，"
          f"wall {wall:.1f}s，{text.strip().splitlines()[-1][:60] if text.strip() else '(no output)'}")

    # 3. fork 炸弹倾向 → 进程数应有上限（mini_sandbox: NPROC=64；容器层另有 PIDs 限制）
    async with session.post(f"{url}/run_code",
                            json={"code": (
                                "import subprocess, sys\n"
                                "procs = []\n"
                                "try:\n"
                                "    for _ in range(500):\n"
                                "        procs.append(subprocess.Popen(['sleep', '30']))\n"
                                "except Exception as e:\n"
                                "    print('blocked at', len(procs), type(e).__name__)\n"
                                "print('spawned', len(procs))\n"
                            ), "language": "python", "run_timeout": 20}) as r:
        status, text, _ = _parse(await _json(r))
    print(f"{'[-]'} fork 探针 → {text.strip().splitlines()[-1][:60] if text.strip() else status}"
          f"（看 spawned 数量判断进程数上限）")

    # 4. 文件系统逃逸 → 写 /tmp 外的位置应受限或至少不污染宿主可见路径
    async with session.post(f"{url}/run_code",
                            json={"code": (
                                "import os\n"
                                "print(os.listdir('/'))\n"
                                "open('/etc/sandbox_probe', 'w').write('x')\n"
                                "print('wrote /etc/sandbox_probe')\n"
                            ), "language": "python", "run_timeout": 10}) as r:
        status, text, _ = _parse(await _json(r))
    escaped = "wrote /etc/sandbox_probe" in text
    print(f"{FAIL if escaped else PASS} 写 /etc → "
          f"{'写入成功（进程级隔离，容器外勿暴露公网！）' if escaped else '被拒绝（有隔离）'}")


# =====================================================================
async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--mode", default="all",
                        choices=["all", "smoke", "latency", "concurrency", "probes"])
    parser.add_argument("--n", type=int, default=30, help="latency 串行次数")
    parser.add_argument("--c", type=int, default=32, help="concurrency 并发数")
    parser.add_argument("--rounds", type=int, default=3, help="concurrency 轮数")
    parser.add_argument("--token", default=None, help="SANDBOX_TOKEN（可选）")
    args = parser.parse_args()

    timeout = aiohttp.ClientTimeout(total=120)
    headers = {"Authorization": f"Bearer {args.token}"} if args.token else {}
    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
        # 连通性
        try:
            async with session.get(f"{args.url}/health") as r:
                if r.status == 200:
                    print(f"{PASS} /health OK（mini_sandbox）")
                else:
                    print(f"[-] /health 返回 {r.status}（sandbox-fusion 没有 /health，忽略）")
        except Exception as exc:
            print(f"{WARN} /health 不可达（{exc}），继续用 /run_code 探测……")

        modes = (["smoke", "latency", "concurrency", "probes"]
                 if args.mode == "all" else [args.mode])
        for mode in modes:
            if mode == "smoke":
                await smoke(session, args.url)
            elif mode == "latency":
                await latency(session, args.url, args.n)
            elif mode == "concurrency":
                await concurrency(session, args.url, args.c, args.rounds)
            elif mode == "probes":
                await probes(session, args.url)


if __name__ == "__main__":
    asyncio.run(main())
