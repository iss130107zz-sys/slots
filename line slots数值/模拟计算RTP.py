import numpy as np
from matplotlib.pyplot import scatter
from numpy import dtype
from openpyxl.styles.builtins import total

"""
复杂3x5老虎机模拟器
- 8种普通符号: 9, 10, J, Q, K, A, Dia, Crown
- 1种Wild (W): 可替代除 Scatter 外的所有符号
- 1种Scatter (S): 触发免费游戏并支付散布赔率
- 25条奖励线
- 免费游戏: 3个S得10次, 4个S得15次, 5个S得20次
  免费游戏中所有赢取×2, 且每个非S非W符号有10%概率变为Wild
- 每局总投注=25单位 (线注=1)
"""

# 滚轮权重
BASE_REEL_WEIGHTS = {
    'reel1_weights' : {'9': 30, '10': 25, 'J': 14, 'Q': 12, 'K': 10,
    'A': 8, 'Dia': 6, 'Crown': 4, 'W': 3, 'S': 2},

    'reel2_weights' : {'9': 30, '10': 25, 'J': 14, 'Q': 12, 'K': 10,
    'A': 8, 'Dia': 6, 'Crown': 4, 'W': 3, 'S': 2},

    'reel3_weights' : {'9': 30, '10': 25, 'J': 14, 'Q': 12, 'K': 10,
    'A': 8, 'Dia': 6, 'Crown': 4, 'W': 3, 'S': 2},

    'reel4_weights' : {'9': 30, '10': 25, 'J': 14, 'Q': 12, 'K': 10,
    'A': 8, 'Dia': 6, 'Crown': 4, 'W': 3, 'S': 2},

    'reel5_weights' : {'9': 30, '10': 25, 'J': 14, 'Q': 12, 'K': 10,
    'A': 8, 'Dia': 6, 'Crown': 4, 'W': 3, 'S': 2}
}

BASE_WEIGHTS = [np.array(list(w.values()),dtype=float) for w in BASE_REEL_WEIGHTS.values()]
BASE_PROBS = [w/ sum(w) for w in BASE_WEIGHTS]


FREE_REEL_WEIGHTS = {
    'reel1_weights' : {'9': 30, '10': 25, 'J': 14, 'Q': 12, 'K': 10,
    'A': 8, 'Dia': 6, 'Crown': 4, 'W': 3, 'S': 2},

    'reel2_weights' : {'9': 30, '10': 25, 'J': 14, 'Q': 12, 'K': 10,
    'A': 8, 'Dia': 6, 'Crown': 4, 'W': 3, 'S': 2},

    'reel3_weights' : {'9': 30, '10': 25, 'J': 14, 'Q': 12, 'K': 10,
    'A': 8, 'Dia': 6, 'Crown': 4, 'W': 3, 'S': 2},

    'reel4_weights' : {'9': 30, '10': 25, 'J': 14, 'Q': 12, 'K': 10,
    'A': 8, 'Dia': 6, 'Crown': 4, 'W': 3, 'S': 2},

    'reel5_weights' : {'9': 30, '10': 25, 'J': 14, 'Q': 12, 'K': 10,
    'A': 8, 'Dia': 6, 'Crown': 4, 'W': 3, 'S': 2}
}

FREE_WEIGHTS = [np.array(list(w.values()),dtype=float) for w in FREE_REEL_WEIGHTS.values()]
FREE_PROBS = [w/ sum(w) for w in FREE_WEIGHTS]




# 赔付表
PAYTABLE = {
    '9':      [5, 10, 50],
    '10':     [8, 20, 80],
    'J':      [10, 30, 120],
    'Q':      [15, 40, 160],
    'K':      [20, 50, 200],
    'A':      [25, 75, 300],
    'Dia':    [40, 100, 400],
    'Crown':  [50, 250, 1000]
}
PAYTABLE_list = []
for val in PAYTABLE.values():
    PAYTABLE_list.append([0,0,0] + val)
PAYTABLE_ARR = np.array(PAYTABLE_list,dtype=float)

SCATTER_PAY_MAP = np.array([0,0,0,2,10,100],dtype=float)
FREE_SPIN_AWARD = {3:10,4:15,5:20}

# 奖励线(转化为一维坐标)
RAW_LINES = [
    [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],  #  0  rows=[0,0,0,0,0] 水平上
    [(0, 0), (0, 1), (1, 2), (2, 3), (2, 4)],  #  1  rows=[0,0,1,2,2] 平→斜下
    [(0, 0), (0, 1), (2, 2), (2, 3), (2, 4)],  #  2  rows=[0,0,2,2,2] 平→跳底
    [(0, 0), (1, 1), (0, 2), (1, 3), (0, 4)],  #  3  rows=[0,1,0,1,0] W形上
    [(0, 0), (1, 1), (1, 2), (1, 3), (0, 4)],  #  4  rows=[0,1,1,1,0] 拱形上
    [(0, 0), (1, 1), (2, 2), (1, 3), (0, 4)],  #  5  rows=[0,1,2,1,0] V形
    [(0, 0), (2, 1), (0, 2), (2, 3), (0, 4)],  #  6  rows=[0,2,0,2,0] 深W上
    [(0, 0), (2, 1), (2, 2), (2, 3), (0, 4)],  #  7  rows=[0,2,2,2,0] 跳底→回
    [(1, 0), (0, 1), (0, 2), (0, 3), (1, 4)],  #  8  rows=[1,0,0,0,1] 上→回中
    [(1, 0), (0, 1), (1, 2), (0, 3), (1, 4)],  #  9  rows=[1,0,1,0,1] 锯齿中
    [(1, 0), (0, 1), (2, 2), (2, 3), (2, 4)],  # 10  rows=[1,0,2,2,2] 上→底
    [(1, 0), (1, 1), (0, 2), (0, 3), (0, 4)],  # 11  rows=[1,1,0,0,0] 中→上
    [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4)],  # 12  rows=[1,1,1,1,1] 水平中
    [(1, 0), (1, 1), (2, 2), (2, 3), (2, 4)],  # 13  rows=[1,1,2,2,2] 中→底
    [(1, 0), (2, 1), (0, 2), (0, 3), (0, 4)],  # 14  rows=[1,2,0,0,0] 下→上
    [(1, 0), (2, 1), (1, 2), (2, 3), (1, 4)],  # 15  rows=[1,2,1,2,1] 深锯齿中
    [(1, 0), (2, 1), (2, 2), (2, 3), (1, 4)],  # 16  rows=[1,2,2,2,1] 拱形下
    [(2, 0), (0, 1), (0, 2), (0, 3), (2, 4)],  # 17  rows=[2,0,0,0,2] 下→上→回
    [(2, 0), (0, 1), (2, 2), (0, 3), (2, 4)],  # 18  rows=[2,0,2,0,2] 锯齿下
    [(2, 0), (1, 1), (0, 2), (1, 3), (2, 4)],  # 19  rows=[2,1,0,1,2] 倒V形
    [(2, 0), (1, 1), (1, 2), (1, 3), (2, 4)],  # 20  rows=[2,1,1,1,2] 拱形下
    [(2, 0), (1, 1), (2, 2), (1, 3), (2, 4)],  # 21  rows=[2,1,2,1,2] 浅锯齿下
    [(2, 0), (2, 1), (0, 2), (0, 3), (0, 4)],  # 22  rows=[2,2,0,0,0] 底→上
    [(2, 0), (2, 1), (1, 2), (0, 3), (0, 4)],  # 23  rows=[2,2,1,0,0] 底斜上
    [(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)]   # 24  rows=[2,2,2,2,2] 水平下
]
FLAT_LINES = np.array([[r*5+c for (r,c) in line]for line in RAW_LINES])


# ===========游戏===========
def calc_base_wins(screens):
    B = screens.shape[0]

    batch_lines = screens[:,FLAT_LINES]

    max_line_pays = np.zeros((B,len(RAW_LINES)))

    for s in range(8):
        is_s = (batch_lines == s)
        is_w = (batch_lines == 8)
        matches = is_s | is_w
        lengths = np.cumprod(matches, axis=2).sum(axis=2)

        # ===== 加这一段 =====
        pos_idx = np.arange(5).reshape(1, 1, 5)  # [0,1,2,3,4]
        in_segment = pos_idx < lengths[..., np.newaxis]  # 哪些位置在winning segment内  [T,T,T,F,F]
        has_s = (is_s & in_segment).any(axis=2)  # segment里是否真的出现了s  [F,F,F,F,F] & [T,T,T,F,F] -> [F,F,F,F,F] -> F
        lengths = np.where(has_s, lengths, 0)  # 全是Wild就不算
        # ===================

        symbol_pay_table = np.array(PAYTABLE_ARR[s])
        max_line_pays = np.maximum(max_line_pays, symbol_pay_table[lengths])



    round_wins = max_line_pays.sum(axis=1)

    # 计算scatter
    scatters = (screens==9).sum(axis=1)
    scatters_wins = np.zeros(B)
    s_mask = scatters >= 3
    s_counts = np.minimum(scatters[s_mask],5)
    scatters_wins[s_mask] = SCATTER_PAY_MAP[s_counts]*len(RAW_LINES)

    return round_wins+scatters_wins,scatters

def run_free_spins(initial_s_count,rng):
    free_spins_remaining = FREE_SPIN_AWARD[initial_s_count]
    fs_win = 0

    while free_spins_remaining > 0:
        free_spins_remaining -= 1

        fs_screen = np.empty(15, dtype=int)
        for c in range(5):
            fs_screen[c]      = rng.choice(10, p=FREE_PROBS[c])
            fs_screen[c + 5]  = rng.choice(10, p=FREE_PROBS[c])
            fs_screen[c + 10] = rng.choice(10, p=FREE_PROBS[c])

        upgrade_mask = (fs_screen!=9) & (fs_screen!=8)
        rand_vals = rng.random(15)
        fs_screen[upgrade_mask & (rand_vals<0.1)] = 8

        win_amt,fs_s_count = calc_base_wins(fs_screen.reshape(1,-1))
        fs_win += win_amt[0]*2

        s_cnt = fs_s_count[0]
        if s_cnt >= 3:
            free_spins_remaining += FREE_SPIN_AWARD[min(s_cnt,5)]
    return fs_win

    return

# ===========模拟===========
def simulate_game(num_games=500000,batch_size=25000):
    rng = np.random.default_rng()


    num_batches = num_games//batch_size
    total_wins_ar = np.array([])
    total_fg_wins_ar = np.array([])

    for b in range(num_batches):
        base_screens = np.empty((batch_size, 15), dtype=int)
        for c in range(5):          # 5 个滚轮
            # 一次生成该 column 的 3 个行样本 (batch_size, 3)
            col_samples = rng.choice(10, size=(batch_size, 3), p=BASE_PROBS[c])
            base_screens[:, c]      = col_samples[:, 0]   # 第 0 行
            base_screens[:, c + 5]  = col_samples[:, 1]   # 第 1 行
            base_screens[:, c + 10] = col_samples[:, 2]   # 第 2 行

        base_wins,scatter_counts = calc_base_wins(base_screens)

        free_game_win = np.zeros_like(base_wins)
        batch_total_wins = base_wins.copy()

        fs_trigger_indices = np.where(scatter_counts>=3)[0]
        for idx in fs_trigger_indices:
            s_count = min(scatter_counts[idx],5)

            free_game_win[idx] += run_free_spins(s_count,rng)
            batch_total_wins[idx] += free_game_win[idx]

        total_fg_wins_ar = np.concatenate([total_fg_wins_ar,free_game_win])
        total_wins_ar = np.concatenate([total_wins_ar,batch_total_wins])

    total_wins = total_wins_ar.sum()
    total_fg_win = total_fg_wins_ar.sum()


    total_bet = num_games * len(RAW_LINES)
    rtp = (total_wins/total_bet) * 100
    fg_rtp = (total_fg_win/total_bet) * 100
    hit_rate = np.mean(total_wins_ar>0)
    game_std = total_wins_ar.std()

    print('\n============统计结果============')
    print(f'总模拟局数：  {num_games}')
    print(f'免费游戏rtp： {fg_rtp:.2f}%')
    print(f'总rtp：   {rtp:.2f}%')
    print(f'中奖率：    {hit_rate*100:.2f}%')
    print(f'标准差   {game_std:.2f}')

if __name__ == "__main__":
    import time

    start = time.time()
    simulate_game(1000000)
    print(f"总耗时: {time.time() - start:.4f} 秒")
