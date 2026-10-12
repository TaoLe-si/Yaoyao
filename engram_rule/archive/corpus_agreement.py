"""主谓一致 + 吸引子干扰语料（把"泛化优势"验到可信规模）。

设计（静态分析）:
  句子 = `the N_head (P the N_attr)×k V` , k∈{0,1,2}
  · **吸引子的单复数与中心词独立随机** ⇒ 约 50% 句子的"最近名词"与中心词不一致
  · 动词必须与**中心词**（不是最近名词）一致 ⇒ 最近名词启发式在吸引子上会答错
  · **未见组合 = 留出整个词项**（2/8 名词 + 1/3 动词对训练里从不出现）
  · 主指标 = 动词单复数正确率, 按 吸引子/无吸引子 分开报

必须验的前提（对应 checks_eval 的 A1–A4）:
  A1 组合空间 ≥ 需求行数（上次中文语料的死循环就是没算这个）
  A2 词项级不相交（旧检查只查行, 会漏掉"没见过的词")
  A3 **最近名词启发式的错误率显著 > 0**（否则任务无位置捷径的区分度）
  A4 所有 while 有硬上限 + 断言
"""
import torch

N_SG = ["dog", "cat", "bird", "horse", "fish", "goat", "pig", "cow"]
N_PL = ["dogs", "cats", "birds", "horses", "fishes", "goats", "pigs", "cows"]
PREP = ["near", "of", "behind", "with"]
V_SG = ["sees", "chases", "likes"]
V_PL = ["see", "chase", "like"]
DET = "the"
END = "."

TOKS = [DET, END] + N_SG + N_PL + PREP + V_SG + V_PL
T = {t: i for i, t in enumerate(TOKS)}
VOCAB = len(TOKS)
PAD = T[END]

HELD_NOUN_IDX = {6, 7}      # goat/pig 一族: 训练里从不出现
HELD_VERB_IDX = {2}         # likes/like 一对: 训练里从不出现


def make_sentence(g, hold_nouns=False, hold_verbs=False, force_attractor=None, max_pp=2):
    """返回 (token 列表, 动词下标, 是否吸引子句)。吸引子 = 最近名词与中心词单复数不同。"""
    if hold_nouns:
        hi = sorted(HELD_NOUN_IDX)[int(torch.randint(0, len(HELD_NOUN_IDX), (1,), generator=g))]
    else:
        # 非留出时避开留出词项 (词项级不相交)
        cand = [i for i in range(len(N_SG)) if i not in HELD_NOUN_IDX]
        hi = cand[int(torch.randint(0, len(cand), (1,), generator=g))]
    head_sg = bool(torch.randint(0, 2, (1,), generator=g))
    toks = [T[DET], T[N_SG[hi] if head_sg else N_PL[hi]]]
    k = int(torch.randint(0, max_pp + 1, (1,), generator=g))   # ★ 可加深 PP 数 (扩距离用)
    last_sg = head_sg
    for _ in range(k):
        p = PREP[int(torch.randint(0, len(PREP), (1,), generator=g))]
        cand2 = [i for i in range(len(N_SG)) if (hold_nouns or i not in HELD_NOUN_IDX)]
        ai = cand2[int(torch.randint(0, len(cand2), (1,), generator=g))]
        asg = bool(torch.randint(0, 2, (1,), generator=g)) if force_attractor is None \
            else (not head_sg if force_attractor else head_sg)
        toks += [T[p], T[DET], T[N_SG[ai] if asg else N_PL[ai]]]
        last_sg = asg
    if hold_verbs:
        vi = sorted(HELD_VERB_IDX)[int(torch.randint(0, len(HELD_VERB_IDX), (1,), generator=g))]
    else:
        cand3 = [i for i in range(len(V_SG)) if i not in HELD_VERB_IDX]
        vi = cand3[int(torch.randint(0, len(cand3), (1,), generator=g))]
    vpos = len(toks)
    toks.append(T[V_SG[vi] if head_sg else V_PL[vi]])
    toks.append(T[END])
    attractor = (k > 0) and (last_sg != head_sg)
    return toks, vpos, attractor, head_sg


def corpus(n_rows, L, seed=0, hold_nouns=False, hold_verbs=False,
           force_attractor=None, avoid=None, budget=None, max_pp=2):
    """生成 n_rows 行; 行级去重 + 硬上限 + 断言 (A4)。返回 (x, vpos, attr, head_sg)"""
    g = torch.Generator().manual_seed(seed)
    seen = set(avoid or ())
    rows, vps, ats, hs = [], [], [], []
    budget = n_rows * 60 + 2000 if budget is None else budget   # 可显式给小预算 (供 A4 验断言)
    while len(rows) < n_rows and budget > 0:
        budget -= 1
        toks, vp, at, h = make_sentence(g, hold_nouns, hold_verbs, force_attractor, max_pp)
        if len(toks) > L:
            continue
        row = tuple(toks + [PAD] * (L - len(toks)))
        if row in seen:
            continue
        seen.add(row); rows.append(list(row)); vps.append(vp); ats.append(at); hs.append(h)
    assert len(rows) == n_rows, "语料只生成 %d/%d 行 (预算耗尽 ⇒ 组合空间不足?)" % (len(rows), n_rows)
    return (torch.tensor(rows, dtype=torch.long), torch.tensor(vps),
            torch.tensor(ats, dtype=torch.bool), torch.tensor(hs, dtype=torch.bool))


@torch.no_grad()
def number_acc(logits, x, vpos, head_sg):
    """动词单复数正确率: 在动词位置, 正确单复数的一组动词的总概率 > 错误一组。
    (比"猜中具体哪个动词"公平: 同单复数有 3 个可换的动词。)"""
    V1 = torch.tensor([T[v] for v in V_SG])
    V2 = torch.tensor([T[v] for v in V_PL])
    lg = logits[torch.arange(x.shape[0]), vpos]                 # (n, vocab)
    p = lg.softmax(-1)
    p_sg = p[:, V1].sum(-1)
    p_pl = p[:, V2].sum(-1)
    correct = torch.where(head_sg, p_sg > p_pl, p_pl > p_sg)
    return correct


def nearest_noun_heuristic(x, vpos, head_sg):
    """A3 用: "动词跟最近的名词一致" 启发式 —— 在吸引子句上必然答错。
    最近名词 = 动词前紧邻的名词 (若无 PP 则就是中心词)。"""
    ok = []
    for i in range(x.shape[0]):
        vp = int(vpos[i])
        # 往前找最近的名词 token
        sg = None
        for j in range(vp - 1, -1, -1):
            t = int(x[i, j])
            if t in [T[n] for n in N_SG]:
                sg = True; break
            if t in [T[n] for n in N_PL]:
                sg = False; break
        ok.append(bool(sg) == bool(head_sg[i]))
    return torch.tensor(ok)
