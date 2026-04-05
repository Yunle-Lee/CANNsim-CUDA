# npu-cuda-bench

终端双栏实时对比 **CANN SIM（Ascend NPU 仿真）** 与 **CUDA（NVIDIA GPU）** 算子性能的 TUI 工具。

支持 33 个数学算子的横向对比，左栏跑华为 CANN 仿真器（dav_3510 / Ascend950），右栏跑 CUDA kernel，结果并排展示。

---

## 效果预览

```

                         ██████╗ █████╗ ███╗  ██╗███╗  ██╗      ██████╗██╗   ██╗██████╗  █████╗
                        ██╔════╝██╔══██╗████╗ ██║████╗ ██║     ██╔════╝██║   ██║██╔══██╗██╔══██╗
                        ██║     ███████║██╔██╗██║██╔██╗██║     ██║     ██║   ██║██║  ██║███████║
                        ╚██████╗██║  ██║██║ ████║██║ ████║     ╚██████╗╚██████╔╝██████╔╝██║  ██║
                         ╚═════╝╚═╝  ╚═╝╚═╝  ╚══╝╚═╝  ╚══╝     ╚═════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝
...
┌─────────────────────────────────┬─────────────────────────────────┐
│  CANN SIM (Ascend950 dav_3510)  │  CUDA (NVIDIA GPU)              │
├─────────────────────────────────┼─────────────────────────────────┤
│  Status : PASS                  │  Status : PASS                  │
│  Wall   : 12.3s                 │  Wall   : 0.1s                  │
│  Ticks  : 4096                  │  Latency: 0.312 µs              │
│  SimTime: 38.4 ms               │  Output : 1.0, 2.0, 3.0, 4.0   │
│  Output : 1.0, 2.0, 3.0, 4.0   │                                 │
└─────────────────────────────────┴─────────────────────────────────┘
── Comparison ──
  NPU ticks    : 4096
  CUDA latency : 0.312 µs
```

---

## 项目结构

```
npu-cuda-bench/
├── bench.py          # 主程序：TUI + 并发调度
└── ops/
    ├── abs.cu        # CUDA abs kernel
    ├── abs_cuda      # 编译好的可执行文件（nvcc 产物）
    ├── add.cu
    ├── add_cuda
    └── ...           # 共 33 个算子的 .cu 源码和编译产物
```

---

## 核心实现思路

### 整体架构

```
TUI 层 (curses)
    └── 并发调度层 (threading)
            ├── NPU 侧：调用 ops-math build.sh --simulator 跑 CANN 仿真
            └── CUDA 侧：直接执行编译好的 CUDA 可执行文件
```

### NPU 侧（CANN SIM）

通过 `subprocess` 调用 `ops-math/build.sh`，传入 `--simulator --soc=ascend950` 参数，让 CANN 工具链在 x86 上用 `dav_3510` 软件仿真器模拟 Ascend950 执行算子：

```python
cmd = f"source {CANN_ENV} && " \
      f"export LD_LIBRARY_PATH={CUSTOM_LIB}:$LD_LIBRARY_PATH && " \
      f"cd {OPS_MATH} && " \
      f"bash build.sh --run_example {op} eager cust " \
      f"--vendor_name=custom --simulator --soc=ascend950 2>&1"
```

从输出中用正则提取关键指标：
- `Total tick: N` — NPU 时钟周期数
- `Model RUN TIME: X ms` — 仿真运行时间
- `result[N] is: X` — 算子输出值（用于正确性验证）

### CUDA 侧

每个算子对应一个独立的 `.cu` 文件，编译为可执行文件。执行时：
1. 先做 10 次 warmup（消除 JIT 编译、缓存冷启动影响）
2. 再跑 100 次，用 `cudaEvent` 精确计时
3. 输出格式固定为 `CUDA_TIME_US=X.XXX`，供 `bench.py` 解析

```c
cudaEventRecord(start);
for (int i = 0; i < iters; i++)
    abs_kernel<<<blocks, threads>>>(d_in, d_out, n);
cudaEventRecord(stop);
cudaDeviceSynchronize();
float ms;
cudaEventElapsedTime(&ms, start, stop);
printf("CUDA_TIME_US=%.3f\n", ms / iters * 1000.0f);
```

### 并发执行

NPU 仿真耗时较长（数十秒），CUDA 执行极快（毫秒级）。两侧在独立线程中并发运行，互不阻塞：

```python
threading.Thread(target=npu_thread, daemon=True).start()
threading.Thread(target=cuda_thread, daemon=True).start()
```

TUI 以 200ms 刷新率轮询状态，运行中显示 `⏳ Running...`，完成后立即展示结果。

---

## 环境要求

### 必须

| 组件 | 版本要求 |
|------|----------|
| Python | 3.8+ |
| CANN 工具链 | 9.0.0-beta.2（含仿真器） |
| ops-math 仓库 | 已编译，路径 `~/workspace/ops-math` |
| NVIDIA GPU | 支持 CUDA 的任意型号 |
| CUDA Toolkit | 11.x 或 12.x（含 nvcc） |

### 可选

```bash
pip install wcwidth   # 终端宽字符对齐（Banner 显示更准确）
```

---

## 复刻步骤

### 第一步：安装 CANN 工具链

从华为昇腾社区下载 CANN 9.0.0-beta.2 安装包：

```bash
chmod +x Ascend-cann-toolkit_9.0.0-beta.2_linux-x86_64.run
./Ascend-cann-toolkit_9.0.0-beta.2_linux-x86_64.run --install --quiet
source /usr/local/Ascend/cann/set_env.sh
echo 'source /usr/local/Ascend/cann/set_env.sh' >> ~/.bashrc
```

安装后确认仿真器存在：

```bash
ls /usr/local/Ascend/cann-9.0.0-beta.2/tools/simulator/dav_3510/
```

### 第二步：编译 ops-math 自定义算子库

```bash
git clone https://gitcode.com/cann/ops-math.git ~/workspace/ops-math
cd ~/workspace/ops-math

# 安装 Python 依赖
pip install -r requirements.txt

# 编译（以 abs 为例，--pkg 生成安装包）
bash build.sh --pkg --soc=ascend950 --ops=abs --vendor_name=custom

# 安装自定义算子包
./build_out/cann-ops-math-custom_linux-x86_64.run --install --quiet
```

编译所有算子（耗时较长）：

```bash
bash build.sh --pkg --soc=ascend950 --vendor_name=custom
```

### 第三步：验证 CANN 仿真可用

```bash
cd ~/workspace/ops-math
bash build.sh --run_example abs eager cust \
    --vendor_name=custom \
    --simulator \
    --soc=ascend950
```

输出中应包含 `execute samples success` 或 `result[0] is: X.X`。

**已知问题（CANN 9.0.0-beta.2 beta 版 bug）：**

若 `cannsim record` 报 `IndentationError`，手动修复：

```bash
# 找到有问题的文件
find /usr/local/Ascend -name "Ascend910_9599_ESL.py" 2>/dev/null
# 用编辑器修复缩进错误后重试
```

若遇到 `aic_reg_read undefined symbol` 错误，说明 cannsim 报告生成不可用，但直接运行仿真（`--simulator`）仍然正常，bench.py 使用的正是直接运行方式。

### 第四步：安装 CUDA Toolkit

```bash
# Ubuntu 示例（根据实际 CUDA 版本调整）
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-x
```

验证：

```bash
nvcc --version
nvidia-smi
```

### 第五步：编译 CUDA 算子

```bash
cd /path/to/npu-cuda-bench/ops

# 编译单个算子
nvcc -O2 abs.cu -o abs_cuda

# 批量编译所有算子
for f in *.cu; do
    op="${f%.cu}"
    nvcc -O2 "$f" -o "${op}_cuda" && echo "✓ $op" || echo "✗ $op"
done
```

### 第六步：修改路径配置

编辑 `bench.py` 顶部的路径变量，确保与实际环境一致：

```python
OPS_MATH   = os.path.expanduser("~/workspace/ops-math")
CANN_ENV   = "/usr/local/Ascend/cann/set_env.sh"
CUSTOM_LIB = "/usr/local/Ascend/cann-9.0.0-beta.2/opp/vendors/custom_math/op_api/lib"
```

### 第七步：运行

```bash
python3 bench.py
```

---

## 操作说明

| 按键 | 功能 |
|------|------|
| `←` / `→` 或 `h` / `l` | 切换算子 |
| `r` | 重新运行当前算子 |
| `q` / `Esc` | 退出 |

切换算子时自动触发重新运行，两侧并发执行。

---

## 支持的算子列表

一元算子（单输入）：
`abs` `acos` `acosh` `asin` `atan` `ceil` `cos` `cosh` `erf` `erfc`
`exp` `expm1` `floor` `log` `log1p` `neg` `reciprocal` `round` `rsqrt`
`sign` `sin` `sinh` `sqrt` `tanh` `trunc`

二元算子（双输入）：
`add` `div` `maximum` `minimum` `mul` `pow` `sub`

---

## 指标说明

| 指标 | 含义 |
|------|------|
| NPU Ticks | Ascend950 仿真时钟周期数（从仿真器日志提取） |
| NPU SimTime | 仿真器报告的模型运行时间（ms） |
| CUDA Latency | 100 次 kernel 执行的平均耗时（µs），warmup 10 次后计时 |
| Wall | 整个命令的实际墙钟时间（含编译、初始化等开销） |

注意：NPU Ticks 是仿真周期数，不能直接与 CUDA µs 做性能比较，两者运行在完全不同的硬件模型上。

---

## 扩展新算子

**添加 CUDA 侧：**

在 `ops/` 目录新建 `<op_name>.cu`，输出格式必须包含：

```c
printf("CUDA_TIME_US=%.3f\n", ms / iters * 1000.0f);
printf("result[0] = %.6f\n", h_out[0]);  // 可选，用于正确性验证
```

然后编译：

```bash
nvcc -O2 ops/<op_name>.cu -o ops/<op_name>_cuda
```

**添加 NPU 侧：**

在 `~/workspace/ops-math/math/` 下按现有算子目录结构添加新算子，参考 `math/abs/` 的 `op_host/`、`op_kernel/`、`examples/` 结构。

**注册到 bench.py：**

在 `bench.py` 的 `OPS` 列表中追加算子名称即可。

---

## 依赖环境汇总

```
# CANN 侧
CANN 9.0.0-beta.2
ops-math（已编译安装）
Ubuntu 22.04 / 24.04 (x86_64)

# CUDA 侧
CUDA Toolkit 11.x / 12.x
NVIDIA GPU（任意支持 CUDA 的型号）

# Python
Python 3.8+
wcwidth（可选）
```

---

## 开源协议

MIT
