# 远端 HTTP 代码沙箱：部署与接入

目标：给 agentic RL 的 tool loop 提供一个**独立于训练进程的隔离执行环境**，
通过 HTTP 调用（`aiohttp` POST `/run_code`），训练进程里不直接 `subprocess` 模型生成的代码。

## 一、威胁模型（决定隔离强度）

要防的是**自家模型产出代码的误伤**：死循环、OOM、fork 炸弹、乱写文件。
不是防真实攻击者。所以进程级隔离（超时 + rlimits）就够起步；
要硬边界时再上容器方案。

## 二、三种部署方式

### 方式 A：mini_sandbox（本仓库自带，推荐起步）

```bash
pip install fastapi uvicorn pydantic
python scripts/mini_sandbox.py --port 8080        # 默认只绑 127.0.0.1
```

- 只有 python 一个运行时，进程级隔离：超时（wall 30s 硬上限）、
  RLIMIT_AS 512MB、RLIMIT_CPU 20s、RLIMIT_NPROC 64、RLIMIT_FSIZE 256MB
- 响应格式刻意做成 sandbox-fusion 兼容（`run_result.stdout/stderr/return_code`），
  下游工具代码两个后端通用
- Windows 也能跑（rlimits 自动跳过，只剩超时隔离）；真训练放 Linux 上
- 可选鉴权：起服务时 `SANDBOX_TOKEN=xxx`，客户端带
  `Authorization: Bearer xxx`

### 方式 B：sandbox-fusion 官方 Docker（有 Docker 的 Linux 机器）

```bash
docker run -d --rm -p 8080:8080 volcengine/sandbox-fusion:server-20250609
# 国内镜像：
docker run -d --rm -p 8080:8080 vemlp-cn-beijing.cr.volces.com/preset-images/code-sandbox:server-20250609
```

- 容器级隔离，是三条路里唯一的硬边界
- 镜像 10+ GB（23 种语言运行时），只要 python 别用这条
- 注意：**AutoDL 实例内没有 Docker**，这条路只适用于有 Docker 的独立机器

### 方式 C：sandbox-fusion 手动安装（AutoDL 可用）

按官方文档 conda + poetry 装，`make run`，只装 python 运行时。
隔离同样是进程级，但资源限额、超时逻辑更完善。

## 三、部署拓扑

| 场景 | 沙箱放哪 | 客户端 URL |
|---|---|---|
| 单机（最常见） | 和训练同机，绑 127.0.0.1 | `http://127.0.0.1:8080` |
| 训练机 busy，沙箱放另一台 | 独立 Linux 机器 | SSH 隧道：`ssh -L 8080:127.0.0.1:8080 user@box`，仍用 `http://127.0.0.1:8080` |
| AutoDL | 实例内手动安装（方式 C）或 mini_sandbox | `http://127.0.0.1:8080` |

**不要把沙箱端口直接暴露公网**。进程级隔离扛不住公网扫描。

## 四、接入训练框架

客户端侧约定（与 verl 教程 SandboxTool 同款契约）：

- `POST {sandbox_url}/run_code`，body：`{"code": ..., "language": "python", "run_timeout": 10}`
- 响应取 `run_result.stdout + run_result.stderr` 作为 tool 返回文本
- 工具类基类的 `create()` 里建 `aiohttp.ClientSession`（复用连接池），
  `release()` 里 close——**不要每次 execute 新建 session**（教程代码是反模式）
- HTTP 客户端超时 = 沙箱 `run_timeout` + 5s 余量，防沙箱本身挂死

```python
class SandboxExecTool:
    async def create(self, instance_id: str, **kw):
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.run_timeout + 5))

    async def execute(self, instance_id: str, parameters: dict, **kw):
        code = extract_code(parameters["code"])       # 剥 ```python 围栏
        async with self._session.post(
            f"{self.sandbox_url}/run_code",
            json={"code": code, "language": "python", "run_timeout": self.run_timeout},
        ) as resp:
            resp.raise_for_status()
            r = (await resp.json())["run_result"]
        text = (r["stdout"] + r["stderr"]).strip() or "(no output)"
        return text, 0.0, {"return_code": r["return_code"]}   # (ToolResponse, reward, metrics)

    async def release(self, instance_id: str, **kw):
        await self._session.close()
```

## 五、会咬人的细节

1. **返回给模型的输出要截断**（服务端已截到 4000 字符）。模型把
   `print(open('bigfile').read())` 塞进上下文，下一轮 prompt 就爆了
2. **`-I` 隔离模式**：mini_sandbox 用 `python -I`（isolated）跑，不带用户
   site-packages。沙箱里缺依赖时，在沙箱服务的解释器环境里装
   （不要为了省事把训练环境挂进去——那等于没有隔离）
3. **run_timeout 别超过工具调用的整体预算**。K 轮轨迹里一次 30s 的
   超时 × batch 32 就是 16 分钟纯等待；python 题目 5~10s 足够
4. **reward 放 trajectory 末端，不放 execute 里**（教程返回的 0.0 就是这个
   含义）。做 turn-level reward（惩罚超时调用、奖励一次通过）以后再改这里
5. **每次执行是全新进程、全新临时目录**，没有状态延续。需要多步有状态
   执行（类 Jupyter）是另一档工程，起步阶段不要做
