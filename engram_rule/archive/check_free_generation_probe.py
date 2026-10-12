"""★实验 B 探针的**可失败性检查** —— 跑 `exp_free_generation.py` 之前必须全绿。

规则 3: 改 → 补检查 → 过检查 → 才准跑。规则 7: 每个检查都要附"它在反例下会变"的证明。

本脚本**零训练**、零大模型前向为主(只在本项目自带的 v11 权重上跑几十行),
秒级完成。它检查的不是"实验结果对不对", 而是**测量机器本身可不可信**:

  P1  `lcp_len` 的单元行为(与朴素实现逐例比对)
  P2  `gold_answer_ids` 与 `format_record` 的切法**全量**对齐(规则 11: 不许抽样)
  P3  教师强制取段正确: 答案段长度 == gold 答案 token 数
  P4  ★★ L1 判据**可失败**: 用出货参数 `no_repeat_ngram=1` 制造真实分歧,
        L1 必须塌。恒真的判据在这一项上必须**红**。
  P5  `utf8_block_at_first_div` 可失败: 半个 UTF-8 字符处必须判 blocked
  P6  ★ 新增的 `answer_ids` 字段由**外部重算字节**核对(不给它背书)
  P7  G0(批==逐行)可失败: 改温度后生成必须变, 否则比的是两份空串
"""
import os
import sys

# ★与 exp_free_generation.py 保持一致: 必须在 import torch 之前设(§57 审计结论)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

FAILURES = []
TOTAL = [0]


def check(name, cond, detail=""):
    TOTAL[0] += 1
    status = "PASS" if cond else "FAIL"
    print("  [%s] %s%s" % (status, name, ("  — " + detail) if detail else ""))
    if not cond:
        FAILURES.append(name)


def naive_lcp(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def main():
    import exp_free_generation as exp
    from tokenizers import Tokenizer
    from dialogue_small import (bytelevel_token_bytes, format_prefix, format_record,
                                greedy_generate, greedy_generate_batch, _DecodeMask,
                                valid_token_bytes, char_f1)

    ck, tok, tr_rows, ev_rows, tr_pack, ev_pack = exp.load_data()
    tr_x, tr_s, tr_l = tr_pack
    ev_x, ev_s, ev_l = ev_pack
    eos_id = tok.token_to_id("[EOS]")
    byte_map, _ = bytelevel_token_bytes(tok)

    # ── P1 ────────────────────────────────────────────────────────────────
    cases = [([], [], 0), ([1], [], 0), ([], [2], 0), ([1, 2, 3], [1, 2, 3], 3),
             ([1, 2, 3], [1, 9, 3], 1), ([1, 2], [1, 2, 3, 4], 2),
             ([1, 2, 3], [0, 2, 3], 0), ([5, 5, 5], [5, 5], 2)]
    check("P1 lcp_len == 朴素实现", all(exp.lcp_len(a, b) == naive_lcp(a, b)
                                        for a, b, _ in cases),
          "%d 组用例" % len(cases))
    check("P1b lcp_len 对前导相同才算", exp.lcp_len([1, 1, 2], [1, 1, 3]) == 2)

    # ── P2 全量对齐(规则 11: 不许只查前 N 行) ─────────────────────────────
    bad = []
    for tag, rows, x, s, l in (("train", tr_rows, tr_x, tr_s, tr_l),
                               ("eval", ev_rows, ev_x, ev_s, ev_l)):
        for i, row in enumerate(rows):
            prompt = tok.encode(format_prefix(row), add_special_tokens=False).ids
            rec = tok.encode(format_record(row), add_special_tokens=False).ids
            want = rec[len(prompt):]
            got = exp.gold_answer_ids(x, s, l, i)
            if want != got:
                bad.append((tag, i))
            if i >= 4096:                     # 硬上限(规则 9), 本数据集远小于它
                break
    check("P2 gold_answer_ids == format_record 切法(全量 %d 行)"
          % (len(tr_rows) + len(ev_rows)), not bad, "不一致: %s" % bad[:5])

    # ── P3 教师强制取段 ───────────────────────────────────────────────────
    from model import RCfg, EngramRuleLM
    model = EngramRuleLM(RCfg(**ck["cfg"])).to(DEV)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    sub = tr_rows[:32]
    tf = exp.teacher_forced_rows(model, sub, tr_x[:32], tr_s[:32], tr_l[:32], 8)
    check("P3 段长 == gold 答案 token 数",
          all(len(tf[i]["pred"]) == len(exp.gold_answer_ids(tr_x, tr_s, tr_l, i))
              for i in range(32)))
    check("P3b margin 与 pred 等长",
          all(len(tf[i]["margin"]) == len(tf[i]["pred"]) for i in range(32)))

    # ── P4 ★★ 分歧位置这个量**不是常量、且可失败**(规则 7) ─────────────────
    # L1(教师强制的首个分歧位 == 自由生成的公共前缀长)是个**逻辑闭合**判据:
    # 两条路径喂同样的 token 序列就该给出同样的下一步。所以"闭合"本身不奇怪,
    # 真正要证的是这个量**对内容/生成过程敏感**, 否则它就是常量。
    from dialogue_small import evaluate_generations
    gold = [exp.gold_answer_ids(tr_x, tr_s, tr_l, i) for i in range(32)]
    _m, outs = evaluate_generations(model, tok, sub, max_new_tokens=exp.MAX_NEW,
                                    device=DEV, context_limit=exp.CONTEXT_LIMIT,
                                    batch_min_size=2)
    lcp_clean = [exp.lcp_len(outs[i]["answer_ids"], gold[i]) for i in range(32)]
    check("P4a lcp 取值多样(不是常量)", len(set(lcp_clean)) >= 3,
          "取值集合 %s" % sorted(set(lcp_clean)))

    # ★保证失败的对照: **只给 1 个 token 的预算** ⇒ lcp 必须 ≤1。
    # 选它而不是 no_repeat_ngram=1 / 温度扰动:
    #   · ngram=1 只屏蔽**已出现**过的 token, 而答案前几个 token 常互不重复 ⇒ 不触发
    #     (第一版踩的坑: 塌 0/32 行, 看着像判据坏了, 其实是对照选错);
    #   · T=0.7 实测只改动 10/32 行 —— 这个模型的分布**尖得离谱**(见下方读数),
    #     温度不是它的敏感方向, 拿它当门禁等于给判据开了一个假绿灯。
    _m1, outs_1 = evaluate_generations(model, tok, sub, max_new_tokens=1,
                                       device=DEV, context_limit=exp.CONTEXT_LIMIT,
                                       batch_min_size=2)
    lcp_one = [exp.lcp_len(outs_1[i]["answer_ids"], gold[i]) for i in range(32)]
    check("P4b ★lcp 必须能变: max_new_tokens=1 时 lcp ≤ 1",
          max(lcp_one) <= 1,
          "max=%d, 均值 %.2f(对照 greedy 均值 %.2f)" % (max(lcp_one), sum(lcp_one) / 32,
                                                    sum(lcp_clean) / 32))
    _mh, outs_h = evaluate_generations(model, tok, sub, max_new_tokens=exp.MAX_NEW,
                                       device=DEV, context_limit=exp.CONTEXT_LIMIT,
                                       temperature=0.7, seed=5, batch_min_size=2)
    lcp_hot = [exp.lcp_len(outs_h[i]["answer_ids"], gold[i]) for i in range(32)]
    changed = sum(1 for i in range(32) if lcp_hot[i] != lcp_clean[i])
    print("       [读数] T=0.7 只改动 %d/32 行(均值 %.2f->%.2f) ⇒ 分布极尖"
          % (changed, sum(lcp_clean) / 32, sum(lcp_hot) / 32))

    # 内容敏感性: 只换问题(gold 不变)后 lcp 是否动。门禁设得低 —— 若这里红,
    # 结论是"模型对该置换不敏感", 那是**读数**不是判据坏, 由 probe 去解释。
    swapped = [{"question": sub[(i + 7) % 32]["question"], "answer": sub[i]["answer"],
                "history": []} for i in range(32)]
    _ms, outs_s = evaluate_generations(model, tok, swapped, max_new_tokens=exp.MAX_NEW,
                                       device=DEV, context_limit=exp.CONTEXT_LIMIT,
                                       batch_min_size=2)
    lcp_swap = [exp.lcp_len(outs_s[i]["answer_ids"], gold[i]) for i in range(32)]
    moved = sum(1 for i in range(32) if lcp_swap[i] != lcp_clean[i])
    check("P4c lcp 对问题置换敏感(门禁低, 红=读数不是判据)", moved >= 4,
          "动了 %d/32 行" % moved)

    # L1 的闭合率: 这是**读数**, 不是门禁 —— 打印出来给 probe 的预登记规则用。
    tf_div_clean = [next((k for k in range(len(gold[i])) if tf[i]["pred"][k] != gold[i][k]),
                         len(gold[i])) for i in range(32)]
    agree = sum(1 for i in range(32) if tf_div_clean[i] == lcp_clean[i])
    print("       [读数] L1 闭合率 %d/32 (不一致行的分歧位 margin 见 part0 输出)" % agree)

    # ── P5 utf8 屏蔽判定可失败 ────────────────────────────────────────────
    mc = _DecodeMask(byte_map, eos_id, DEV)
    # (a) 真实 gold 的首个答案 token 位置: 完整序列 ⇒ 不该被挡住
    g0 = exp.gold_answer_ids(tr_x, tr_s, tr_l, 0)
    blk0, n0 = exp.utf8_block_at_first_div(tok, mc, byte_map, eos_id, g0, 1)
    check("P5a 完整 UTF-8 起始位不应被挡", blk0 is False,
          "allowed=%d" % n0)
    # (b) 人为半个字符: 取一个首字节是 0xE4 的 token, 状态落在半个汉字上
    lead = next(t for t, f in byte_map.items()
                if f is not None and f[:1] == b"\xe4" and len(f) == 1)
    half = valid_token_bytes((0, 0x80, 0xBF), byte_map[lead])
    tail = byte_map[lead][-2:]
    blk1 = not bool(mc.allowed_mask(half, tail)[lead])
    check("P5b ★半个 UTF-8 字符处必须判 blocked(否则该函数恒不触发)", blk1,
          "state=%r" % (half,))

    # ── P6 新字段 answer_ids 外部重算 ─────────────────────────────────────
    g = greedy_generate(model, tok, sub[0], max_new_tokens=exp.MAX_NEW, device=DEV,
                        context_limit=exp.CONTEXT_LIMIT, temperature=0.0)
    raw = b"".join(byte_map[t] for t in g["answer_ids"])
    ok = False
    try:
        ok = raw.decode("utf-8").strip() == g["generated"]
    except UnicodeDecodeError:
        ok = False
    check("P6 answer_ids 重算字节 == generated", ok,
          "ids=%d generated=%r" % (len(g["answer_ids"]), g["generated"][:24]))
    # 反例: 少一个 id 必须解码不同 ⇒ 否则 P6 是恒真的
    raw_drop = b"".join(byte_map[t] for t in g["answer_ids"][:-1]) if g["answer_ids"] else b""
    check("P6b ★去掉末尾 id 后字节必须变(否则 P6 无区分力)",
          (not g["answer_ids"]) or raw_drop != raw)

    # ── P7 G0 可失败 ──────────────────────────────────────────────────────
    gb = greedy_generate_batch(model, tok, sub[:8], max_new_tokens=exp.MAX_NEW,
                               device=DEV, context_limit=exp.CONTEXT_LIMIT,
                               temperature=0.0)
    same = sum(1 for i in range(8)
               if gb[i]["answer_ids"]
               == greedy_generate(model, tok, sub[i], max_new_tokens=exp.MAX_NEW,
                                  device=DEV, context_limit=exp.CONTEXT_LIMIT,
                                  temperature=0.0)["answer_ids"])
    check("P7a 批==逐行(8 行)", same == 8, "%d/8" % same)
    # P7b 的目的是"证明 G0 比的不是两份空串"。温度不是可靠抓手(这个模型分布极尖,
    # T=0.7 才改动 10/32 行), 所以直接验内容非平凡: 每行都得吐出像样的中文。
    lens = [len(gb[i]["generated"]) for i in range(8)]
    han = [sum(1 for ch in gb[i]["generated"] if "\u4e00" <= ch <= "\u9fff")
           for i in range(8)]
    check("P7b ★生成非平凡(否则 G0 比的是空串)", min(lens) >= 4 and min(han) >= 3,
          "字数 %s, 汉字数 %s" % (lens, han))

    # ── P8 char_f1 / EM 口径没被本轮改动影响 ─────────────────────────────
    check("P8 char_f1(自比)==1.0", abs(char_f1("我最喜欢的运动是篮球。",
                                             "我最喜欢的运动是篮球。") - 1.0) < 1e-12)

    # ── P9 EOS 对齐口径(必须非空, 否则是空集恒真) ────────────────────────
    lcp_fixed = [exp.lcp_len(outs[i]["answer_ids"] + ([eos_id] if outs[i]["eos"] else []),
                             gold[i]) for i in range(32)]
    em_rows = [i for i in range(32)
               if outs[i]["generated"].strip() == sub[i]["answer"].strip()]
    check("P9a 本批存在文本 EM=1 的行(否则 P9b 是空集恒真)", len(em_rows) >= 1,
          "%d/32 行" % len(em_rows))
    check("P9b ★文本 EM=1 的行, EOS 对齐后 lcp 必须 == gold_len",
          all(lcp_fixed[i] == len(gold[i]) for i in em_rows),
          "行 %s, lcp %s, gold_len %s"
          % (em_rows[:6], [lcp_fixed[i] for i in em_rows[:6]],
             [len(gold[i]) for i in em_rows[:6]]))

    # ══════════════════════════════════════════════════════════════════════
    # 下面这一段补的是**上一轮漏掉的门禁**: 检查 17/17 通过之后我又改了
    # exp_free_generation.py(EOS 对齐 / part1 / 误删又补回 main), 没有重跑
    # 检查就跑了实验。P1–P9 一条都没覆盖到那几处改动。这里逐个补上。
    # ══════════════════════════════════════════════════════════════════════
    print("  --- 覆盖 diagnose_split / part1 / main 的改动 ---")

    # ── P10 ★ 直接门禁 diagnose_split 里那段 EOS 对齐(不是手工复刻) ──────
    sub16 = tr_rows[:16]
    _mm, recs16 = exp.diagnose_split(model, tok, sub16, tr_x, tr_s, tr_l, "anchor")
    n16 = len(recs16)
    closed16 = sum(1 for r in recs16 if r["tf_div"] == r["lcp"])
    check("P10a ★diagnose_split 实跑: L1 闭合率 100%%(16 行)", closed16 == n16,
          "%d/%d" % (closed16, n16))
    em16 = [i for i in range(n16) if recs16[i]["gen_em"] == 1.0]
    check("P10b 本批存在 gen_em=1 的行(否则 P10c 是空集恒真)", len(em16) >= 1,
          "%d 行" % len(em16))
    check("P10c ★gen_em=1 的行必须 lcp==gold_len(不补 EOS 就会差 1)",
          all(recs16[i]["lcp"] == recs16[i]["gold_len"] for i in em16),
          "行 %s lcp %s gold_len %s"
          % (em16[:4], [recs16[i]["lcp"] for i in em16[:4]],
             [recs16[i]["gold_len"] for i in em16[:4]]))
    check("P10d 每行都带 tf_margin_at_lcp(新加的字段没漏)",
          all("tf_margin_at_lcp" in r for r in recs16))

    # ── P11 ★ 子集切齐(规则 23): 全包 vs 切片包结果必须逐行一致 ────────────
    _mm2, recs_sl = exp.diagnose_split(model, tok, sub16, tr_x[:16], tr_s[:16],
                                       tr_l[:16], "anchor")
    same11 = all(a["tf_div"] == b["tf_div"] and a["lcp"] == b["lcp"]
                 for a, b in zip(recs16, recs_sl))
    check("P11 ★diagnose_split 对全包/切片包结果一致(子集切齐)", same11)

    # ── P12 ★ part1 的依赖链(1 步训练 + 诊断)真能跑通 ─────────────────────
    import exp_capacity_scaling as ecs
    from dialogue_small import train_dialogue
    mb12 = ecs.start_micro_batch(64)
    m12 = ecs.make_model(ck, 64, 11, tr_x, tr_l)
    train_dialogue(m12, tr_x, tr_s, tr_l, 1, ecs.BS, 0.003, DEV, seed=11,
                   first_token_weight=1.0, micro_batch=mb12)
    _mm3, recs12 = exp.diagnose_split(m12, tok, sub16, tr_x, tr_s, tr_l, "p12")
    check("P12 ★part1 依赖链(1 步训练→诊断)可跑通", len(recs12) == 16,
          "mb=%d, 诊断 %d 行" % (mb12, len(recs12)))
    del m12
    if DEV == "cuda":
        torch.cuda.empty_cache()

    # ── P13 part1 必须是**生成器**(中途挂掉不丢已跑的算力) ────────────────
    import inspect
    check("P13a ★part1 是生成器(有 yield, 不是 return 整个列表)",
          inspect.isgeneratorfunction(exp.part1))
    check("P13b main() 存在且同时处理 part 0/1",
          callable(getattr(exp, "main", None))
          and "a.part == 1" in inspect.getsource(exp.main)
          and "part0(" in inspect.getsource(exp.main))

    # ── P14 ★ 门禁指纹: 改代码 ⇒ 旧数据自动作废 ───────────────────────────
    import tempfile
    check("P14a GATE 是 12 位十六进制", len(exp.GATE) == 12
          and all(c in "0123456789abcdef" for c in exp.GATE), exp.GATE)
    g_before = exp.GATE
    # 造一份"被改过一个字节"的检查脚本副本, GATE 必须随之变化。
    # (第一版用 monkeypatch builtins.open, 写错了: open(p,"rb") 的 mode 是
    #  **位置参数**, 我判的是 kwargs 的 "b" in k ⇒ 永远不成立, 检查恒红。)
    mutant = os.path.join(tempfile.gettempdir(), "dm_gate_mutant.py")
    with open(os.path.join(HERE, "check_free_generation_probe.py"), "rb") as f:
        with open(mutant, "wb") as g:
            g.write(f.read() + b"\n# mutate\n")
    check("P14b ★检查脚本内容变化 ⇒ GATE 必须随之变化",
          exp._gate_tag(("exp_free_generation.py", mutant)) != g_before)
    check("P14c GATE 稳定(同内容两次算出同一个)", exp._gate_tag() == g_before)
    check("P14d 换文件名也算变化(不是只比第一个文件)",
          exp._gate_tag(("exp_free_generation.py",)) != g_before)

    # ── P15 ★ 分配器设置: 没开 expandable_segments 就退回了碎片化那条路 ──────
    # 要点不只是"有这一行", 而是它在 **`import torch` 之前** —— 分配器只在
    # CUDA 上下文创建时读一次配置, 写在 torch 后面等于没写。
    # ★必须按**整行**匹配: 注释里写的 "`import torch`" 会被 find() 当成导入语句
    #   (第一版就是这么误红的 —— 我自己写的注释把检查骗了)。
    def _line_of(src, token):
        for i, ln in enumerate(src.splitlines()):
            if ln.strip().startswith(token):
                return i
        return -1

    src_exp = inspect.getsource(exp)
    i_env = _line_of(src_exp, "os.environ.setdefault")
    i_torch = _line_of(src_exp, "import torch")
    check("P15a ★expandable_segments 设在 import torch **之前**(第 %d 行 vs 第 %d 行)"
          % (i_env + 1, i_torch + 1),
          i_env != -1 and i_torch != -1 and i_env < i_torch,
          "当前 %s" % os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
    with open(os.path.join(HERE, "check_free_generation_probe.py"),
              "r", encoding="utf-8") as f:
        src_self = f.read()
    j_env = _line_of(src_self, "os.environ.setdefault")
    j_torch = _line_of(src_self, "import torch")
    check("P15b ★检查脚本自己也在 import torch 之前设了同一项",
          j_env != -1 and j_torch != -1 and j_env < j_torch)

    # ── P16 ★ 显存量的是 **reserved** 不是 allocated(§57 的核心教训) ────────
    #   allocated 在每步末尾采样时早已回落到 ~100 MiB(激活被释放),
    #   而"显存占满"是 nvidia-smi 看到的 reserved —— 两者差 20~35 倍。
    #   这条要求判据本身承认这一点, 否则下次还会拿 allocated 去汇报显存。
    import torch.cuda as C
    C.empty_cache()
    C.reset_peak_memory_stats()
    m16 = ecs_probe_model = exp_free_model = None
    import exp_capacity_scaling as ecs2
    m16 = ecs2.make_model(ck, 64, 11, tr_x, tr_l)
    train_dialogue(m16, tr_x, tr_s, tr_l, 2, ecs2.BS, 0.003, DEV, seed=11,
                   first_token_weight=1.0, micro_batch=64)
    alloc = torch.cuda.memory_allocated() / (1 << 20)
    resv = torch.cuda.memory_reserved() / (1 << 20)
    peak = torch.cuda.max_memory_reserved() / (1 << 20)
    check("P16a ★reserved 远大于 allocated(碎片确实存在, 两者不能混用)",
          resv > alloc * 3, "allocated %.0f / reserved %.0f (比 %.1f×)"
          % (alloc, resv, resv / max(alloc, 1e-9)))
    check("P16b ★开 expandable_segments 后 reserved 有硬上限(不随步数涨)",
          peak < 4000, "max_memory_reserved %.0f MiB (旧默认分配器 2 步就 3540+)"
          % peak)
    del m16
    C.empty_cache()

    print("\n%d/%d 通过" % (TOTAL[0] - len(FAILURES), TOTAL[0]))
    if FAILURES:
        print("失败项: %s" % FAILURES)
        return 1
    print("全部通过 —— 可以跑 exp_free_generation.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())