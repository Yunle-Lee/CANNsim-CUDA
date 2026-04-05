#!/usr/bin/env python3
"""
npu-cuda-bench: 终端双栏对比 CANN SIM vs CUDA 算子性能
"""
import curses, subprocess, re, os, time, threading

OPS_MATH = os.path.expanduser("~/workspace/ops-math")
BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
CANN_ENV = "/usr/local/Ascend/cann/set_env.sh"
CUSTOM_LIB = "/usr/local/Ascend/cann-9.0.0-beta.2/opp/vendors/custom_math/op_api/lib"

# ── CANN SIM runner ───────────────────────────────────────────────────────────

def run_npu(op: str) -> dict:
    cmd = f"source {CANN_ENV} && " \
          f"export LD_LIBRARY_PATH={CUSTOM_LIB}:$LD_LIBRARY_PATH && " \
          f"cd {OPS_MATH} && " \
          f"bash build.sh --run_example {op} eager cust " \
          f"--vendor_name=custom --simulator --soc=ascend950 2>&1"
    t0 = time.time()
    out = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True).stdout
    elapsed = time.time() - t0

    result = {"op": op, "status": "FAIL", "wall_s": elapsed}
    m = re.search(r"Total tick:\s*(\d+)", out)
    if m:
        result["ticks"] = int(m.group(1))
    m = re.search(r"Model RUN TIME:\s*([\d.]+)\s*ms", out)
    if m:
        result["sim_ms"] = float(m.group(1))
    m = re.search(r"block_start.*?tick=(\d+).*?block_end.*?tick=(\d+)", out, re.DOTALL)
    if not m:
        m = re.search(r"\[block_start\].*?core_id=(\d+).*?\[block_end\].*?core_id=\1", out, re.DOTALL)
    # 提取结果验证
    results = re.findall(r"result\[\d+\] is: ([\d.]+)", out)
    if results:
        result["outputs"] = results[:4]
        result["status"] = "PASS"
    elif "execute samples success" in out:
        result["status"] = "PASS"
    result["raw_tail"] = out[-500:] if len(out) > 500 else out
    return result

# ── CUDA runner ───────────────────────────────────────────────────────────────

def build_cuda(op: str) -> bool:
    src = os.path.join(BENCH_DIR, "ops", f"{op}.cu")
    out = os.path.join(BENCH_DIR, "ops", f"{op}_cuda")
    if not os.path.exists(src):
        return False
    r = subprocess.run(["nvcc", "-O2", src, "-o", out], capture_output=True)
    return r.returncode == 0

def run_cuda(op: str, n: int = 8) -> dict:
    exe = os.path.join(BENCH_DIR, "ops", f"{op}_cuda")
    if not os.path.exists(exe):
        if not build_cuda(op):
            return {"op": op, "status": "BUILD_FAIL"}
    t0 = time.time()
    r = subprocess.run([exe, str(n)], capture_output=True, text=True)
    elapsed = time.time() - t0
    out = r.stdout
    result = {"op": op, "status": "FAIL", "wall_s": elapsed}
    m = re.search(r"CUDA_TIME_US=([\d.]+)", out)
    if m:
        result["time_us"] = float(m.group(1))
        result["status"] = "PASS"
    results = re.findall(r"result\[\d+\] = ([\d.]+)", out)
    if results:
        result["outputs"] = results[:4]
    return result

# ── TUI ───────────────────────────────────────────────────────────────────────

BANNER = [
    r" ██████╗ █████╗ ███╗  ██╗███╗  ██╗      ██████╗██╗   ██╗██████╗  █████╗ ",
    r"██╔════╝██╔══██╗████╗ ██║████╗ ██║     ██╔════╝██║   ██║██╔══██╗██╔══██╗",
    r"██║     ███████║██╔██╗██║██╔██╗██║     ██║     ██║   ██║██║  ██║███████║",
    r"╚██████╗██║  ██║██║ ████║██║ ████║     ╚██████╗╚██████╔╝██████╔╝██║  ██║",
    r" ╚═════╝╚═╝  ╚═╝╚═╝  ╚══╝╚═╝  ╚══╝     ╚═════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝",
]

C_TITLE  = 1
C_PASS   = 2
C_FAIL   = 3
C_DIM    = 4
C_BOLD   = 5
C_CYAN   = 6
C_YELLOW = 7

def init_colors():
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(C_TITLE,  curses.COLOR_CYAN,   -1)
    curses.init_pair(C_PASS,   curses.COLOR_GREEN,  -1)
    curses.init_pair(C_FAIL,   curses.COLOR_RED,    -1)
    curses.init_pair(C_DIM,    curses.COLOR_WHITE,  -1)
    curses.init_pair(C_BOLD,   curses.COLOR_WHITE,  -1)
    curses.init_pair(C_CYAN,   curses.COLOR_CYAN,   -1)
    curses.init_pair(C_YELLOW, curses.COLOR_YELLOW, -1)

def draw(scr, op, npu_res, cuda_res, npu_running, cuda_running, ops_list=None, op_idx=0):
    scr.erase()
    h, w = scr.getmaxyx()
    half = w // 2

    def safe(y, x, text, attr=0):
        try: scr.addstr(y, x, text[:w-x-1], attr)
        except curses.error: pass

    # Banner：CANN 红色，CUDA 绿色，每行按中间分割
    try:
        from wcwidth import wcswidth
        dispw = wcswidth
    except ImportError:
        dispw = len
    if w >= 40:
        for i, line in enumerate(BANNER):
            lw = dispw(line)
            x = max(0, (w - lw) // 2)
            # 每行大约前半是 CANN，后半是 CUDA（以空格分隔两个单词）
            mid = len(line) // 2
            left = line[:mid]
            right = line[mid:]
            try:
                scr.addstr(i + 1, x, left, curses.color_pair(C_FAIL) | curses.A_BOLD)
                scr.addstr(i + 1, x + dispw(left), right, curses.color_pair(C_PASS) | curses.A_BOLD)
            except curses.error:
                pass
        row0 = len(BANNER) + 1
    else:
        safe(0, 2, "CANN-CUDA Bench", curses.color_pair(C_TITLE) | curses.A_BOLD)
        row0 = 1

    # 算子选择栏
    if ops_list:
        x = 2
        for i, o in enumerate(ops_list):
            attr = curses.color_pair(C_TITLE) | curses.A_BOLD if i == op_idx else curses.color_pair(C_DIM)
            label = f"[{o}]" if i == op_idx else f" {o} "
            try: scr.addstr(row0, x, label, attr)
            except curses.error: pass
            x += len(label) + 1
        safe(row0, x, "  ←→ switch  r rerun  q quit", curses.color_pair(C_DIM))
    row0 += 1

    safe(row0, 0, "─" * (w-1), curses.color_pair(C_DIM))
    row0 += 1

    # 列标题
    safe(row0, 2,        "CANN SIM (Ascend950 dav_3510)", curses.color_pair(C_CYAN) | curses.A_BOLD)
    safe(row0, half + 2, "CUDA (NVIDIA GPU)",             curses.color_pair(C_YELLOW) | curses.A_BOLD)
    row0 += 1
    safe(row0, 0, "─" * (half-1), curses.color_pair(C_DIM))
    safe(row0, half, "│", curses.color_pair(C_DIM))
    safe(row0, half+1, "─" * (w-half-2), curses.color_pair(C_DIM))
    row0 += 1

    row = 4

    def show_col(x, res, running):
        nonlocal row
        r = row
        if running:
            safe(r, x, "  ⏳ Running...", curses.color_pair(C_YELLOW))
            return
        if not res:
            safe(r, x, "  (not started)", curses.color_pair(C_DIM))
            return
        status_attr = curses.color_pair(C_PASS) if res.get("status") == "PASS" else curses.color_pair(C_FAIL)
        safe(r,   x, f"  Status : {res.get('status','?')}", status_attr | curses.A_BOLD); r+=1
        safe(r,   x, f"  Wall   : {res.get('wall_s',0):.1f}s", curses.color_pair(C_DIM)); r+=1
        if "ticks" in res:
            safe(r, x, f"  Ticks  : {res['ticks']}", curses.color_pair(C_BOLD)); r+=1
        if "sim_ms" in res:
            safe(r, x, f"  SimTime: {res['sim_ms']:.1f} ms", curses.color_pair(C_BOLD)); r+=1
        if "time_us" in res:
            safe(r, x, f"  Latency: {res['time_us']:.3f} µs", curses.color_pair(C_BOLD)); r+=1
        if "outputs" in res:
            safe(r, x, f"  Output : {', '.join(res['outputs'][:4])}", curses.color_pair(C_DIM)); r+=1

    # 左列 NPU
    row = row0
    show_col(1, npu_res, npu_running)
    # 右列 CUDA
    row = row0
    for rr in range(row0, row0 + 6):
        safe(rr, half, "│")
    show_col(half + 2, cuda_res, cuda_running)

    # 分隔线
    row = max(row, row0 + 6)
    safe(row, 0, "─" * (w-1), curses.color_pair(C_DIM))
    row += 1

    # 对比摘要
    if npu_res and cuda_res and npu_res.get("status") == "PASS" and cuda_res.get("status") == "PASS":
        safe(row, 2, "── Comparison ──", curses.color_pair(C_TITLE) | curses.A_BOLD); row += 1
        if "ticks" in npu_res:
            safe(row, 4, f"NPU ticks     : {npu_res['ticks']}", curses.color_pair(C_CYAN)); row += 1
        if "time_us" in cuda_res:
            safe(row, 4, f"CUDA latency  : {cuda_res['time_us']:.3f} µs", curses.color_pair(C_YELLOW)); row += 1
        safe(row, 4, "Note: NPU ticks = clock cycles on Ascend950 sim", curses.color_pair(C_DIM)); row += 1
        safe(row, 4, "      CUDA time = avg over 100 kernel launches",   curses.color_pair(C_DIM)); row += 1

    # footer
    safe(h-1, 0, " q:quit  r:rerun ", curses.color_pair(C_DIM))
    scr.refresh()

def main(scr):
    init_colors()
    curses.curs_set(0)
    scr.nodelay(True)
    scr.timeout(200)

    OPS = [
        "abs","acos","acosh","asin","atan","ceil","cos","cosh",
        "erf","erfc","exp","expm1","floor","log","log1p","neg",
        "reciprocal","round","rsqrt","sign","sin","sinh","sqrt",
        "tanh","trunc","add","div","maximum","minimum","mul","pow","sub",
    ]
    op_idx = 0
    op = OPS[op_idx]
    npu_res = None
    cuda_res = None
    npu_running = False
    cuda_running = False

    def run_both():
        nonlocal npu_res, cuda_res, npu_running, cuda_running
        npu_res = None
        cuda_res = None
        npu_running = True
        cuda_running = True

        def npu_thread():
            nonlocal npu_res, npu_running
            npu_res = run_npu(op)
            npu_running = False

        def cuda_thread():
            nonlocal cuda_res, cuda_running
            cuda_res = run_cuda(op)
            cuda_running = False

        threading.Thread(target=npu_thread, daemon=True).start()
        threading.Thread(target=cuda_thread, daemon=True).start()

    run_both()

    while True:
        draw(scr, op, npu_res, cuda_res, npu_running, cuda_running, OPS, op_idx)
        key = scr.getch()
        if key in (ord('q'), ord('Q'), 27):
            break
        elif key in (ord('r'), ord('R')):
            run_both()
        elif key in (curses.KEY_LEFT, ord('h')) and not (npu_running or cuda_running):
            op_idx = (op_idx - 1) % len(OPS)
            op = OPS[op_idx]
            run_both()
        elif key in (curses.KEY_RIGHT, ord('l')) and not (npu_running or cuda_running):
            op_idx = (op_idx + 1) % len(OPS)
            op = OPS[op_idx]
            run_both()

if __name__ == "__main__":
    curses.wrapper(main)
