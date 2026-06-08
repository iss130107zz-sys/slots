import numpy as np
from time import perf_counter

'''

'''
# ============================================================
# 1. 常量定义
# ============================================================

# 网格规格: 6列 × 5行，共 30 格
ROWS, COLS = 5, 6
CELLS = ROWS * COLS

# 编码方案
#   0         空位（消除后占位）
#   1 - 9     9 种普通符号（4 种糖果 + 5 种水果）
#   10        Scatter（棒棒糖）
#   11 - 26   16 种倍率符号（仅免费旋转中出现）
EMPTY = 0
REGULAR_START, REGULAR_END = 1, 9
SCATTER = 10
MULT_START, MULT_END = 11, 26
N_MULTIPLIERS = MULT_END - MULT_START + 1

# 符号名称（仅用于可读性，不影响模拟逻辑）
SYMBOL_NAMES = [
    "",
    "红糖果", "粉糖果", "绿糖果", "蓝糖果",    # 1-4  四种糖果
    "苹果", "桃子", "西瓜", "葡萄", "香蕉",    # 5-9  五种水果
    "SCATTER",                               # 10   Scatter
]

# 倍率符号对应倍率值（16 种，索引 11-26）
MULTIPLIER_VALUES = np.array(
    [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 50, 100, 500, 1000, 2500],
    dtype=np.float64,
)

# 普通符号赔付表（按投注的倍数）
#   行 = 符号 1..9（红 粉 绿 蓝 苹 桃 西 葡 蕉）
#   列 = 档位:  0 → 8-9 个   1 → 10-11 个   2 → 12 个及以上
PAYTABLE = np.array([
    [10,   25,   50  ],   # 红糖果
    [2.5,  10,   25  ],   # 粉糖果
    [2,    5,    15  ],   # 绿糖果
    [1.5,  2,    12  ],   # 蓝糖果
    [1,    1.5,  10  ],   # 苹果
    [0.8,  1.2,  8   ],   # 桃子
    [0.5,  1,    5   ],   # 西瓜
    [0.4,  0.9,  4   ],   # 葡萄
    [0.25, 0.75, 2   ],   # 香蕉
], dtype=np.float64)

# Scatter 赔付: 数量 → 投注倍数（基础游戏 & 免费旋转通用）
#   4 个 = 3x    5 个 = 5x    6 个及以上 = 100x
SCATTER_PAY = {4: 3.0, 5: 5.0, 6: 100.0}


# ============================================================
# 权重设定
# ============================================================

# 基础游戏权重（符号 1-9 + scatter 10）
#   低价值符号（高赔付）权重小 → 出现少
#   高价值符号（低赔付）权重大 → 出现频繁
_BASE_RAW = np.array(
    [4.0, 7.0, 8.5, 10.5, 11.5, 13.0, 18.0, 19.0, 24.0, 3.0],  # 红 粉 绿 蓝 苹 桃 西 葡 蕉 + scatter
    dtype=np.float64,
)
BASE_WEIGHTS = _BASE_RAW / _BASE_RAW.sum()
_BASE_INDS = np.arange(1, 11, dtype=int)

# 免费旋转权重（符号 1-9 + scatter + 倍率符号 11-26）
#   普通符号部分：权重略低于基础游戏，scatter 权重降低
_FREE_REG = np.array(
    [3.5, 4.5, 5.5, 6.5, 10.0, 11.0, 11.5, 11.5, 12.5, 2.0],  # 红 粉 绿 蓝 苹 桃 西 葡 蕉 + scatter
    dtype=np.float64,
)
# 倍率符号权重：与倍率值成反比
#   2x 倍率权重最高（≈1.0），2500x 倍率权重极低（≈0.0008）
_FREE_MULT = 2.0 / MULTIPLIER_VALUES
_FREE_RAW = np.concatenate([_FREE_REG, _FREE_MULT])
FREE_WEIGHTS = _FREE_RAW / _FREE_RAW.sum()
_FREE_INDS = np.arange(1, 1 + len(_FREE_RAW), dtype=int)

print("Constants defined successfully.")
print(f"Base weights: {BASE_WEIGHTS}")
print(f"Free weights shape: {FREE_WEIGHTS.shape}")

# ============================================================
# 2. 核心函数
# ============================================================

def generate_symbols(count, include_multiplier=False, rng=None):
    """生成 count 个随机符号 (一维数组)。"""
    if count <= 0:
        return np.array([], dtype=int)
    if rng is None:
        rng = np.random
    if include_multiplier:
        return rng.choice(_FREE_INDS, size=count, p=FREE_WEIGHTS)
    else:
        return rng.choice(_BASE_INDS, size=count, p=BASE_WEIGHTS)


def generate_grid(include_multiplier=False, rng=None):
    """生成 5x6 初始网格。"""
    return generate_symbols(CELLS, include_multiplier=include_multiplier, rng=rng).reshape(ROWS, COLS)


def detect_payout(grid):
    """
    检测中奖, 返回 (总赔付倍率, win_mask)。

    仅检测普通符号（1-9）中奖：网格上同符号数量 >= 8 即中奖。
    win_mask 标记中奖符号所在位置，供 cascade 消除用。
    Scatter 赔付在 cascade 循环结束后单独计算，不在此处理。
    倍率符号（11-26）永不参与中奖判定，也不会出现在 win_mask 中。
    """
    payout = 0.0
    win_mask = np.zeros_like(grid, dtype=bool)

    # 普通符号 1-9: 计数 >=8 的中奖
    for sym in range(REGULAR_START, REGULAR_END + 1):
        count = np.count_nonzero(grid == sym)
        if count >= 8:
            if count >= 12:
                tier = 2
            elif count >= 10:
                tier = 1
            else:
                tier = 0
            payout += PAYTABLE[sym - 1, tier]
            win_mask |= (grid == sym)

    # Scatter 赔付在外部（spin_base / spin_free / simulate）单独处理
    return payout, win_mask


def cascade(grid, win_mask, include_multiplier=False, rng=None):
    """
    消除中奖符号 → 逐列重力下落 → 顶部填充新符号。
    倍率符号 (11-26) 不会出现在 win_mask 中, 但参与重力下落。
    原地修改 grid 并返回。
    """
    if rng is None:
        rng = np.random

    grid[win_mask] = EMPTY

    for col in range(COLS):
        col_data = grid[:, col]
        non_zero = col_data[col_data != EMPTY]
        kept = len(non_zero)
        if kept < ROWS:
            new_syms = generate_symbols(ROWS - kept, include_multiplier=include_multiplier, rng=rng)
            grid[:, col] = np.concatenate([new_syms, non_zero])
        else:
            grid[:, col] = non_zero

    return grid


def count_scatter(grid):
    """统计 scatter 数量。"""
    return np.count_nonzero(grid == SCATTER)


def sum_multipliers(grid):
    """计算网格上所有倍率符号的倍率之和。"""
    mult_mask = (grid >= MULT_START) & (grid <= MULT_END)
    if not mult_mask.any():
        return 0.0
    indices = grid[mult_mask] - MULT_START
    return MULTIPLIER_VALUES[indices].sum()


print("Core functions defined.")

# ============================================================
# 3. 游戏逻辑
# ============================================================

def spin_base(bet=1.0, rng=None):
    """
    执行一局基础游戏。

    流程:
    1. 生成初始 6×5 网格（无倍率符号）
    2. 循环: 检测中奖 → 累加赔付 → 消除 → 重力下落补新符号，直到无新中奖
    3. 统计 scatter 数量，计算 scatter 赔付（4=3x, 5=5x, 6+=100x）
    4. 4 个及以上 scatter 触发免费旋转

    返回: (总赔付金额, 是否触发免费旋转)
    """
    if rng is None:
        rng = np.random
    grid = generate_grid(include_multiplier=False, rng=rng)
    total_payout = 0.0

    # 消除循环：持续消除中奖符号并 cascade，直到无新中奖
    while True:
        payout, win_mask = detect_payout(grid)
        if payout == 0.0:
            break
        total_payout += payout
        cascade(grid, win_mask, include_multiplier=False, rng=rng)

    # 消除循环结束后统计 scatter 并计算赔付
    sc = count_scatter(grid)
    free_triggered = sc >= 4
    if sc >= 6:
        total_payout += SCATTER_PAY[6]
    elif sc >= 5:
        total_payout += SCATTER_PAY[5]
    elif sc >= 4:
        total_payout += SCATTER_PAY[4]

    return total_payout * bet, free_triggered

def spin_free(bet=1.0, rng=None):
    """
    执行一次免费旋转。

    与基础游戏的区别:
    - 网格中可能出现倍率符号（2x ~ 2500x）
    - 消除循环结束后，总赔付 × 网格上所有倍率符号之和
    - 3 个及以上 scatter 额外奖励 5 次免费旋转
    - scatter 赔付同样适用（4=3x, 5=5x, 6+=100x）

    返回: (总赔付金额, 额外奖励的免费旋转次数 0 或 5)
    """
    if rng is None:
        rng = np.random
    grid = generate_grid(include_multiplier=True, rng=rng)
    total_payout = 0.0

    # 消除循环（包含倍率符号的 cascade）
    while True:
        payout, win_mask = detect_payout(grid)
        if payout == 0.0:
            break
        total_payout += payout
        cascade(grid, win_mask, include_multiplier=True, rng=rng)

    # 统计 scatter：3+ 奖励 5 次额外旋转，4+ 同时有 scatter 赔付
    sc = count_scatter(grid)
    extra_spins = 5 if sc >= 3 else 0
    if sc >= 6:
        total_payout += SCATTER_PAY[6]
    elif sc >= 5:
        total_payout += SCATTER_PAY[5]
    elif sc >= 4:
        total_payout += SCATTER_PAY[4]

    # 倍率符号求和，总赔付乘以倍率和
    mult_sum = sum_multipliers(grid)
    if mult_sum > 0:
        total_payout *= mult_sum

    return total_payout * bet, extra_spins

def run_free_spins(bet=1.0, rng=None):
    """完整免费游戏回合: 初始 10 次 + 每次 3+ scatter 追加 5 次。"""
    if rng is None:
        rng = np.random
    remaining = 10
    total = 0.0

    while remaining > 0:
        remaining -= 1
        payout, extra = spin_free(bet, rng)
        total += payout
        remaining += extra

    return total


print("Game logic defined.")

# ============================================================
# 4. 批量模拟
# ============================================================

def simulate(n, bet=1.0, seed=None):
    """
    批量模拟 n 局 Sweet Bonanza。

    策略:
    1. 一次性生成 n 个 (5,6) 初始网格
    2. 向量化预筛选: 初始网格既无赢奖且 scatter<4 的直接归零
    3. 仅对"活局"跑消除循环
    4. 触发免费旋转的局额外跑免费游戏
    """
    rng = np.random.default_rng(seed)
    t0 = perf_counter()

    # --- 步骤 1: 生成所有初始网格 ---
    print(f"[1/3] 生成 {n:,} 个初始网格 ...", end=" ", flush=True)
    grids = rng.choice(_BASE_INDS, size=(n, ROWS, COLS), p=BASE_WEIGHTS).astype(np.int32)
    print(f"done ({perf_counter() - t0:.2f}s)")

    # --- 步骤 2: 向量化预筛选（纯 numpy，无 Python 循环）---
    print(f"[2/3] 预筛选活局 ...", end=" ", flush=True)
    t1 = perf_counter()

    # 任一普通符号在初始网格中计数 >= 8
    has_win = np.zeros(n, dtype=bool)
    for sym in range(REGULAR_START, REGULAR_END + 1):
        counts = np.count_nonzero(grids == sym, axis=(1, 2))
        has_win |= (counts >= 8)

    # scatter >= 4（触发免费旋转的最低条件）
    scatter_counts = np.count_nonzero(grids == SCATTER, axis=(1, 2))
    scatter_hit = scatter_counts >= 4

    live_mask = has_win | scatter_hit
    live_count = live_mask.sum()
    print(f"done ({perf_counter() - t1:.2f}s), {live_count:,}/{n:,} 活局 ({live_count/n*100:.1f}%)")

    # --- 步骤 3: 跑活局消除循环 ---
    print(f"[3/3] 跑消除循环 ...", end=" ", flush=True)
    t2 = perf_counter()

    payouts = np.zeros(n, dtype=np.float64)
    payouts_free = np.zeros(n, dtype=np.float64)
    free_triggered = np.zeros(n, dtype=bool)


    live_indices = np.where(live_mask)[0]
    report_every = max(1, live_count // 20)  # 每 5% 报告一次

    for i, idx in enumerate(live_indices):
        grid = grids[idx].copy()
        total_payout = 0.0
        ft = False

        # 消除循环：基础游戏无倍率符号
        while True:
            payout, win_mask = detect_payout(grid)
            if payout == 0.0:
                break
            total_payout += payout

            cascade(grid, win_mask, include_multiplier=False, rng=rng)

        # 消除循环结束后统计 scatter 并计算赔付、判定免费触发
        sc = count_scatter(grid)
        if not ft and  sc>= 4:
            ft = True
            if sc >= 6:
                total_payout += SCATTER_PAY[6]
            elif sc >= 5:
                total_payout += SCATTER_PAY[5]
            elif sc >= 4:
                total_payout += SCATTER_PAY[4]

        payouts[idx] = total_payout * bet
        free_triggered[idx] = ft

        # 触发免费旋转 → 跑完整免费游戏（初始 10 次 + 可能追加）
        if ft:
            payouts_free[idx] += run_free_spins(bet, rng)
            payouts[idx] += payouts_free[idx]
        # 进度报告
        if (i + 1) % report_every == 0:
            pct = (i + 1) / live_count * 100
            elapsed = perf_counter() - t2
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"\n      进度 {pct:.0f}% ({i+1:,}/{live_count:,}), {rate:.0f} 局/秒", end="", flush=True)

    elapsed3 = perf_counter() - t2
    print(f"\n      消除循环完成 ({elapsed3:.2f}s, {live_count/elapsed3:.0f} 局/秒)")

    # --- 汇总统计 ---
    total_bet = n * bet
    total_return = payouts.sum()
    rtp = total_return / total_bet * 100           # 总 RTP
    free_game_rtp = payouts_free.sum() / total_bet * 100  # 免费游戏贡献的 RTP

    results = {
        "n": n,
        "bet": bet,
        "total_bet": total_bet,
        "total_return": total_return,
        "rtp": rtp,
        "free_game_rtp": free_game_rtp,
        "hit_rate": (payouts > 0).mean() * 100,
        "free_trigger_rate": free_triggered.mean() * 100,
        "mean_payout": payouts.mean(),
        "median_payout": np.median(payouts),
        "max_payout": payouts.max(),
        "payouts": payouts,
        "elapsed": perf_counter() - t0,
    }
    return results


def print_results(r):
    """格式化打印结果。"""
    print()
    print("=" * 56)
    print(f"{'Sweet Bonanza 2500 模拟结果':^56}")
    print("=" * 56)
    print(f"  模拟局数:     {r['n']:>12,}")
    print(f"  单注金额:     {r['bet']:>12.2f}")
    print(f"  总投注:       {r['total_bet']:>12,.2f}")
    print(f"  总返还:       {r['total_return']:>12,.2f}")
    print(f"  RTP:          {r['rtp']:>11.4f}%")
    print(f"  free_game_RTP: {r['free_game_rtp']:>11.4f}%")
    print(f"  中奖率:       {r['hit_rate']:>11.2f}%")
    print(f"  免费触发率:   {r['free_trigger_rate']:>10.2f}%")
    print(f"  平均奖金:     {r['mean_payout']:>12.4f}")
    print(f"  中位数奖金:   {r['median_payout']:>12.4f}")
    print(f"  最大奖金:     {r['max_payout']:>12.2f}")
    print(f"  总耗时:       {r['elapsed']:>11.2f}s")
    print("=" * 56)


print("Batch simulation defined. Ready to run.")
print(print_results(simulate(10000000)))
