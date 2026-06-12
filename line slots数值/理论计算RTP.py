import numpy as np
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

# # ==============slots data==============

PAYTABLE = {
    '9':      [5, 10, 50],
    '10':     [8, 20, 80],
    'J':      [10, 30, 120],
    'Q':      [15, 40, 160],
    'K':      [20, 50, 200],
    'A':      [25, 75, 300],
    'Dia':    [40, 100, 400],
    'Crown':  [50, 250, 1000],
}

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

SCATTER_PAY_MAP = np.array([0,0,0,2,10,100],dtype=float)
FREE_SPIN_AWARD = {3:10,4:15,5:20}






# ==============base rtp==============
BASE_WEIGHTS = np.array([list(w.values()) for w in BASE_REEL_WEIGHTS.values()],dtype=float)
BASE_PROBS = BASE_WEIGHTS / BASE_WEIGHTS.sum(axis=1,keepdims=True)

np_BASE_PROBS_no_scatter = BASE_PROBS[:,:-1]
np_BASE_PROBS_no_scatter[:,:-1] += np_BASE_PROBS_no_scatter[:,-1].reshape(5,1)
np_BASE_PROBS_no_scatter = np_BASE_PROBS_no_scatter.T


def calc_rtp(np_weight):
    at_least_3 = np_weight[:, :3].cumprod(axis=1)[:, -1]
    at_least_4 = np_weight[:, :4].cumprod(axis=1)[:, -1]
    at_least_5 = np_weight[:, :5].cumprod(axis=1)[:, -1]


    at_least_3[:-1] -= np_weight[-1, :3].cumprod()[-1]
    at_least_4[:-1] -= np_weight[-1, :4].cumprod()[-1]
    at_least_5[:-1] -= np_weight[-1, :5].cumprod()[-1]


    exact_5 = at_least_5
    exact_4 = at_least_4 - at_least_5
    exact_3 = at_least_3 - at_least_4

    p_trigger = np.zeros((8,3))
    p_trigger[:,0] = exact_3[:-1]
    p_trigger[:,1] = exact_4[:-1]
    p_trigger[:,2] = exact_5[:-1]

    np_pay = np.array(list(PAYTABLE.values()))
    return p_trigger * np_pay


# ==============scatter rtp expect==============
BASE_PROBS_scatter = BASE_PROBS[:,-1].tolist()

def scatter_p(scatter_list_weight):
    res = np.array([1.0])
    for p in scatter_list_weight:
        q = 1-p
        poly = np.array([q,p])
        res = np.convolve(res,poly)
    return res

scatter_p_list = scatter_p(BASE_PROBS_scatter*3)

np_scatter_p = np.array([scatter_p_list[3],scatter_p_list[4],np.sum(scatter_p_list[5:])])
np_scatter_rtp = np_scatter_p * SCATTER_PAY_MAP[3:]


base_RTP = calc_rtp(np_BASE_PROBS_no_scatter).sum() + np_scatter_rtp.sum()
print(f'基础RTP：{base_RTP:.2%}')

expect_FG = (np_scatter_p * np.array(list(FREE_SPIN_AWARD.values()))).sum()

# ==============free game rtp==============
FREE_WEIGHTS =np.array([list(w.values()) for w in FREE_REEL_WEIGHTS.values()])
FREE_WEIGHTS_sum = FREE_WEIGHTS.sum(axis=1)
FREE_WEIGHTS_change = np.zeros(FREE_WEIGHTS.shape)
FREE_WEIGHTS_change[:,:-2] = FREE_WEIGHTS[:,:-2] * 0.9
FREE_WEIGHTS_change[:,-1] = FREE_WEIGHTS[:,-1]
FREE_WEIGHTS_change[:,-2] = FREE_WEIGHTS_sum - FREE_WEIGHTS_change.sum(axis=1)
FREE_PROBS = FREE_WEIGHTS_change.T / FREE_WEIGHTS_sum
FREE_PROBS_no_scatter = FREE_PROBS[:-1]
FREE_PROBS_no_scatter[:-1] += FREE_PROBS_no_scatter[-1]

FG_scatter_plist = scatter_p((FREE_WEIGHTS_change[:,-1]/FREE_WEIGHTS_sum).tolist()*3)
np_FG_scatter_p = np.array([FG_scatter_plist[3],FG_scatter_plist[4],sum(FG_scatter_plist[5:])])
FG_np_scatter_rtp = np_FG_scatter_p * SCATTER_PAY_MAP[3:]
FG_expanding = 1/ (1 - (np_FG_scatter_p * np.array(list(FREE_SPIN_AWARD.values()))).sum())



FG_RTP = expect_FG * FG_expanding * (calc_rtp(FREE_PROBS_no_scatter).sum() +FG_np_scatter_rtp.sum()) * 2
print(f'免费游戏RTP：{FG_RTP:.2%}')

print(f'总RTP：{FG_RTP+base_RTP:.2%}')

