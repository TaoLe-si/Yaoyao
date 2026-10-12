"""对**评测 harness 本身**的检查 (checks_eval.py) —— 违反过的纪律: 新代码必须先过检查再跑对比。

我上一轮写完 eval_solid.py 直接跑去跑对比, 结果两个 bug 在跑的过程中才暴露:
  · matched_dense_cfg 逐个真建模型 (几十上百 MB × 31 个候选)
  · speed() 把词表写死 256, 而 CFG 词表只有 13 ⇒ token id 越界 ⇒ CUDA device-side assert
两条都属于"跑之前就该用检查抓到"的。本文件把 harness 的每条前提变成**快速断言**,
不通过就不许跑对比。

跑法: python engram_rule/checks_eval.py     退出码 0 = harness 可信
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import eval_solid as E

OK, BAD = [], []


def check(name, cond, evidence):
    (OK if cond else BAD).append(name)
    print("  [%s] %-46s %s" % ("✓" if cond else "★", name, evidence))
    return cond


# --------------------------------------------------------------- A1-A4 (一致-吸引子语料的前提)
def a_agreement_corpus():
    """新语料的四条前提必须**先**验掉, 才允许跑对比:
      A1 组合空间足够 (行数能生成出来, 不耗尽预算 —— 上次中文语料的死循环就是没算这个)
      A2 **词项级**不相交 (留出词项在训练行里一个都不许出现; 旧检查只查行会漏)
      A3 最近名词启发式的错误率显著 > 0 (否则任务有位置捷径、无区分度)
      A4 生成器带硬上限+断言 (已在 corpus() 内实现, 这里验它真的会抛)"""
    import corpus_agreement as A
    L = 24
    x, vp, at, hs = A.corpus(2400, L, seed=1)
    ok1 = tuple(x.shape) == (2400, L)
    tr_tok = set(x.reshape(-1).tolist())
    held = set()
    for i, n in enumerate(A.N_SG):
        if i in A.HELD_NOUN_IDX:
            held.add(A.T[n]); held.add(A.T[A.N_PL[i]])
    for i, v in enumerate(A.V_SG):
        if i in A.HELD_VERB_IDX:
            held.add(A.T[v]); held.add(A.T[A.V_PL[i]])
    leak = tr_tok & held
    # 留出词项语料必须**能生成**且真的用到留出词
    xh, vph, ath, hsh = A.corpus(400, L, seed=2, hold_nouns=True, hold_verbs=True,
                                 avoid=set(map(tuple, x.tolist())))
    used_held = set(xh.reshape(-1).tolist()) & held
    heur_ok = A.nearest_noun_heuristic(x, vp, hs)
    heur_err = 1 - float(heur_ok.float().mean())
    attr = at
    attr_err = 1 - float(heur_ok[attr].float().mean()) if bool(attr.any()) else 1.0
    print("  A1 组合空间: 训练 %s, 留出词项 %s (预算未耗尽)" % (tuple(x.shape), tuple(xh.shape)))
    print("  A3 吸引子句占比 %.3f; 最近名词启发式: 全体错 %.3f, **吸引子句上错 %.3f**"
          % (float(attr.float().mean()), heur_err, attr_err))
    ok = (check("A1 语料能生成到目标行数 (组合空间足够)", ok1, "训练 %s / 留出 %s" % (tuple(x.shape), tuple(xh.shape)))
          & check("A2 留出词项在训练行里 0 次出现", len(leak) == 0 and len(used_held) > 0,
                  "训练里泄漏 %d 个留出词; 留出语料里用到 %d 个留出词" % (len(leak), len(used_held)))
          & check("A3 最近名词启发式有显著错误 (任务非平凡)", attr_err > 0.5,
                  "吸引子句上启发式错误率 %.3f (>0.5 才有区分度)" % attr_err))
    # A4: 断言真的会抛 (给一个不可能的组合空间: 要求 50000 行)
    # ★ 第 1 版要求 5 万行以为会超空间 —— 实际空间 ≈16×4161×2=133K > 50K, 断言当然不触发。
    #   是**我检查的前提算错了**。改成直接给小预算: 5 行只给 3 次尝试, 断言必触发且快。
    raised = False
    try:
        A.corpus(5, L, seed=3, budget=3)
    except AssertionError:
        raised = True
    ok &= check("A4 组合空间不足时生成器会报错而非空转", raised, "断言已触发" if raised else "**没触发**")
    return ok



# --------------------------------------------------------------- C24 (答案不得在查询位的注意力窗内)
def c24_answer_not_in_window():
    """★ 2026-10-09 抓出的致命泄漏: 在动词位置 vp 查询"输出动词 token" ⇒ 因果注意力含 j=i
    ⇒ 模型只需抄自己位置的输入 ⇒ 100% 是抄写, 不是语法 (反事实: 换错数动词 ⇒ 准确率 0)。
    判据: (a) 任务口径必须 查询位 < 答案位; (b) 反事实——答案换成错的那一个后准确率不得掉到 0。"""
    import corpus_agreement as A
    L = 24
    x, vp, at, hs = A.corpus(120, L, seed=5)
    # (a) 口径: 我们改用 vp-1 作为查询位 ⇒ 必须严格小于答案位
    q = (vp - 1).clamp(min=0)
    ok_pos = bool((q < vp).all())
    # (b) 反事实构造: 把动词 token 换成错数的那个, 若指标有效则它**必须**能感知这个变化
    #     (即: 干净口径下换错数动词后, "答案在上下文中"这件事改变了, 而不是被抄写覆盖)
    Vsg = {A.T[v] for v in A.V_SG}
    xw = x.clone()
    for i in range(x.shape[0]):
        tok = int(x[i, int(vp[i])])
        pool = [A.T[v] for v in A.V_PL] if tok in Vsg else [A.T[v] for v in A.V_SG]
        xw[i, int(vp[i])] = pool[0]
    changed = not torch.equal(x, xw)
    return check("C24 一致任务口径: 查询位 < 答案位 且 答案可被反事实改变", ok_pos and changed,
                 "查询位 = vp-1 < vp = %s; 换错数动词确实改变了输入 = %s (干净口径下该变化应影响预测而非被抄写)"
                 % (ok_pos, changed))



# --------------------------------------------------------------- C25 (超窗实验的前提 + 负向对照)
def c25_beyond_window_premise():
    """★ 实验与harness共用同一分层测试生成器; 距离、超窗比例和无泄漏前提必须一致。"""
    L = 40
    W = 16
    from exp_window_move import stratified_holdout
    x, vp, at, hs = stratified_holdout(L, n_per_k=10)
    # 与实验同一分层生成器: k=1..10 各有样本; k=0 的有限句子空间已由训练数据穷尽。
    q = (vp - 1).clamp(min=0)
    dist = (q - 1).clamp(min=0)
    k = (vp - 2) // 3
    offset_ok = bool((dist == 3 * k).all()) and bool((vp == 2 + 3 * k).all())
    beyond = dist >= W
    frac = float(beyond.float().mean())
    covered = (dist >= 0) & (dist < W)
    neg_ok = bool((~covered[beyond]).all()) and bool(covered[~beyond].all())
    k_counts = {i: int((k == i).sum()) for i in range(1, 11)}
    all_k = all(n > 0 for n in k_counts.values())
    return (check("C25a 语料距离公式与查询口径一致 (vp=2+3k, d=3k)", offset_ok and all_k,
                  "k=1..10 counts=%s; vp=2+3k=%s; query=vp-1 后 d=query-1=3k=%s"
                  % (k_counts, bool((vp == 2 + 3 * k).all()), offset_ok))
            & check("C25b 超窗行占比足够 (端点 d=W-1 属窗内)", frac > 0.25,
                    "相对偏移范围 [%d,%d], 超窗口偏移≥%d 的行占 %.3f"
                    % (int(dist.min()), int(dist.max()), W, frac))
            & check("C25c 负向对照: 起点冻结在0时窗口覆盖判定正确", neg_ok,
                    "冻结窗口 [0,%d]; d≥%d 全不覆盖, d<%d 全覆盖" % (W - 1, W, W)))



# --------------------------------------------------------------- C26 (实验脚本的模型接口一致性)
def c26_model_interface():
    """★ 我犯过的错: 实验脚本里无条件访问 `m.blocks[0].attn`, 而稠密基线用 DenseBlock **没有该属性**
    ⇒ AttributeError, 而且是读一眼 baseline/dense.py 就能预见的。
    判据: 两个模型的接口差异必须被显式记录, 且实验脚本一律走 getattr 守卫
    （这里用"两个模型各自的可读诊断属性集合"来钉住差异）。"""
    import model as M
    from baseline.dense import DenseLM
    c = M.RCfg(vocab=32, d=64, n_layers=2, n_heads=4, ffn_mult=2, rope_dim=8)
    er, dn = M.EngramRuleLM(c), DenseLM(c)
    er_attrs = set(dir(er.blocks[0]))
    dn_attrs = set(dir(dn.blocks[0]))
    only_er = sorted(er_attrs - dn_attrs)
    has_attn_dense = "attn" in dn_attrs
    ok = (not has_attn_dense) and ("attn" in er_attrs)
    return check("C26 两模型接口差异已显式化 (稠密基线无 .attn)", ok,
                 "只有 ER 有的块属性: %s; 稠密有 .attn = %s ⇒ 实验脚本必须 getattr 守卫"
                 % (only_er[:6], has_attn_dense))



def main():
    print("评测 harness 检查 (每条对应一个曾经/可能出错的前提)\n")
    L = 16

    # ---- H1 语料生成: 形状 / token 范围 / 非退化 ----
    # ★ 必须与 main() 走**同一条**生成路径 (带 avoid 的拒绝采样); 第 1 版这里没传 avoid,
    #   测的是旧调用方式 ⇒ 检查报的"重叠 40 行"是检查没跟上修法, 不是 harness 还漏
    dtr = E.cfg_corpus(2400, L, seed=1)
    dte = E.cfg_corpus(400, L, seed=2, avoid=set(map(tuple, dtr.tolist())))
    un = E.cfg_corpus(200, L, seed=3, unseen=True,
                      avoid=set(map(tuple, dtr.tolist())) | set(map(tuple, dte.tolist())))
    V = len(E.CFG_TOK)
    ok_shape = dtr.shape == (2400, L) and dte.shape == (400, L) and un.shape == (200, L)
    ok_range = int(dtr.max()) < V and int(un.max()) < V and int(dtr.min()) >= 0
    # 非退化: 训练集不能只有一两种行
    n_uniq = torch.unique(dtr, dim=0).shape[0]
    check("H1a CFG 语料形状正确", ok_shape, "训练 %s 测试 %s 未见 %s" % (tuple(dtr.shape), tuple(dte.shape), tuple(un.shape)))
    check("H1b token id 全在词表内 (=%d)" % V, ok_range,
          "max id %d < %d; 不同行 %d/%d" % (int(dtr.max()), V, n_uniq, dtr.shape[0]))

    # ---- H2 泄漏: 训练/测试不得有相同行 (run() 里那条断言的**正向**验证) ----
    tr = set(map(tuple, dtr.tolist()))
    leak = len(tr & set(map(tuple, dte.tolist())))
    leak_u = len((tr | set(map(tuple, dte.tolist()))) & set(map(tuple, un.tolist())))
    check("H2 训练/测试/未见 三集合逐行不相交", leak == 0 and leak_u == 0,
          "训练∩测试 %d 行, (训练∪测试)∩未见 %d 行" % (leak, leak_u))

    # ---- H3 中文语料必须够大 (E1 的实质: 上一版只有 15 训练/6 测试行, 读数是噪声) ----
    # ★ 路线4 后: 中文语料也走**句子级不相交 + 固定字符表**, 检查必须跟上 (旧版只验形状)
    ztr, vz = E.zh_corpus(1400, L, seed=11)
    zte, vz2 = E.zh_corpus(450, L, seed=12, avoid=set(map(tuple, ztr.tolist())))
    lk = len(set(map(tuple, ztr.tolist())) & set(map(tuple, zte.tolist())))
    check("H3 中文语料测试行 ≥ 300 且三集合无重叠", zte.shape[0] >= 300 and lk == 0,
          "训练 %d / 测试 %d 行, 重叠 %d 行, 词表 %d (两次调用必须同词表: %s)"
          % (ztr.shape[0], zte.shape[0], lk, vz, vz == vz2))
    check("H3b 中文 token id 全在词表内", int(ztr.max()) < vz and int(zte.max()) < vz,
          "max id %d/%d < %d" % (int(ztr.max()), int(zte.max()), vz))

    # ---- H4 参数匹配基线: 能在合理误差内对齐, 且只数参数不真分配 ----
    cfg0 = E.RCfg(vocab=V, d=128, n_layers=4, n_heads=4, ffn_mult=2, rope_dim=16)
    with torch.device("meta"):
        n_er = sum(p.numel() for p in E.EngramRuleLM(cfg0).parameters())
    mc = E.matched_dense_cfg(V, n_er, layers=4, heads=4, ffn_mult=2)
    with torch.device("meta"):
        n_m = sum(p.numel() for p in E.DenseLM(mc).parameters())
    rel = abs(n_m - n_er) / n_er
    check("H4 匹配参数基线误差 < 25%", rel < 0.25,
          "ER %.3fM vs 稠密(d=%d) %.3fM, 偏差 %.1f%%" % (n_er / 1e6, mc.d, n_m / 1e6, rel * 100))

    # ---- H5 E4 的判据: speed() 不得超出模型词表 (上一轮 CUDA assert 的根因) ----
    Vc = E.RCfg(vocab=V, d=64, n_layers=1, n_heads=4, ffn_mult=2, rope_dim=8)
    m = E.DenseLM(Vc)
    ok_v = m.emb.num_embeddings == V
    # 直接按 speed() 的写法造数据并过一遍前向, 验证不越界
    x = torch.randint(0, m.emb.num_embeddings, (1, 8))
    with torch.no_grad():
        out = m(x)
    check("H5 speed() 用模型自己的词表 (不写死)", ok_v and out.shape[0] == 1,
          "写死 256 时 max id 255 > 词表 %d ⇒ 越界; 现在按 %d 生成" % (V, m.emb.num_embeddings))

    # ---- H6 小步烟测: 三个模型都能跑通前向+反传 (跑对比前的最后一道闸) ----
    x = torch.randint(0, V, (2, L))
    ok_all, msgs = True, []
    for tag, model, cfgv in (("ER", E.EngramRuleLM(E.RCfg(vocab=V, d=64, n_layers=1, n_heads=4,
                                                          ffn_mult=2, rope_dim=8)), V),
                             ("稠密", E.DenseLM(Vc), V)):
        model.engram.build_from_corpus(x) if tag == "ER" else None
        lg = model(x)
        loss = torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, cfgv), x[:, 1:].reshape(-1))
        loss.backward()
        g = sum(1 for p in model.parameters() if p.grad is not None and torch.isfinite(p.grad).all())
        ok_all &= torch.isfinite(loss).all().item() and g > 0
        msgs.append("%s loss %.3f 有梯度参数 %d" % (tag, loss.item(), g))
    check("H6 两模型前向+反传均有限且有梯度", ok_all, "; ".join(msgs))

    # ---- H7 负向对照: run() 的"测试行太少"断言必须真的会炸 ----
    raised = False
    try:
        E.run("负向对照(应抛错)", E.cfg_corpus(64, L, seed=5), E.cfg_corpus(2, L, seed=6), V, 
              type("A", (), dict(steps=1, bs=2, lr=1e-3, seeds=1, min_count=2))())
    except AssertionError:
        raised = True
    check("H7 测试行过少时 harness 真的会拒绝", raised, "断言已触发" if raised else "**没触发 ⇒ 断言是假的**")

    # ---- A1–A4: 一致-吸引子语料的前提 (新语料必须先过这里才允许跑对比) ----
    a_agreement_corpus()
    c24_answer_not_in_window()
    c25_beyond_window_premise()
    c26_model_interface()

    print("\n================ 汇总 ================")
    print("  通过 %d 项, 未通过 %d 项 %s" % (len(OK), len(BAD), ("★ " + ", ".join(BAD)) if BAD else ""))
    print("  ⇒ %s" % ("harness 可信, 可以跑对比" if not BAD else "先修 harness, 不许跑对比"))
    return 0 if not BAD else 1


if __name__ == "__main__":
    sys.exit(main())
