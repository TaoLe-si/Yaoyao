"""Engram-Rule 训练 + 指导书 §7 的指标台, 并与稠密基线 (baseline/dense.py) A/B。

指标: 困惑度 / 下一词准确率 / 推理 tokens/s (与基线比) / 稀疏度 / 加速比上界 /
      Engram 表规模 / 可解释性 (导出规则) / 域外泛化 (留出 n-gram 段)。
数据: Zipf 合成语料 (长尾, 与真实语料的 n-gram 分布同形); 前 80% 训练, 后 20% 留出。
"""
import argparse, os, sys, time, tempfile, shutil
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
from model import RCfg, EngramRuleLM
from baseline.dense import DenseLM

DEV = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_SEED = 7
EVAL_SEED = 7


def zipf_data(cfg, n_tok, seed=0, alpha=1.1):
    g = torch.Generator().manual_seed(seed)
    p = torch.arange(1, cfg.vocab + 1, dtype=torch.float32).pow(-alpha)
    p /= p.sum()
    return torch.multinomial(p, n_tok, replacement=True, generator=g)


def batches(data, bs, L, n, gen):
    for _ in range(n):
        i = torch.randint(0, len(data) - L - 1, (bs,), generator=gen)
        idx = i[:, None] + torch.arange(L + 1)[None, :]
        yield data[idx[:, :-1]].to(DEV), data[idx[:, 1:]].to(DEV)


def evaluate(model, data_te, L, bs, cfg, seed=7):
    gen = torch.Generator().manual_seed(seed)
    model.eval()
    with torch.no_grad():
        tot = n = acc = 0.0
        for x, y in batches(data_te, bs, L, 8, gen):
            lg = model(x)
            l = F.cross_entropy(lg.reshape(-1, cfg.vocab), y.reshape(-1))
            tot += l.item() * y.numel(); n += y.numel()
            acc += (lg.argmax(-1) == y).float().sum().item()
    ppl = __import__("math").exp(min(tot / n, 20))
    x = next(iter(batches(data_te, 1, L, 1, gen)))[0]
    with torch.no_grad():
        for _ in range(3): model(x)
        if DEV == "cuda": torch.cuda.synchronize()
        t1 = time.time()
        for _ in range(10): model(x)
        if DEV == "cuda": torch.cuda.synchronize()
        tok_s = 10 * L / (time.time() - t1)
    return dict(ppl=ppl, acc=acc / n, tok_s=tok_s, params=sum(p.numel() for p in model.parameters()))


def validate_training_data(data_tr, L, bs):
    starts = len(data_tr) - L - 1
    if L < 2 or bs < 1 or starts < 1:
        raise ValueError("训练流长度 %d、序列长 %d、batch %d 不合法 (至少需要一个采样起点)"
                         % (len(data_tr), L, bs))


def validate_eval_data(data_te, L):
    if L < 2 or len(data_te) - L - 1 < 1:
        raise ValueError("评测流长度 %d 不足以形成序列长 %d 的窗口" % (len(data_te), L))


def train_model(model, data_tr, L, steps, bs, lr, tag, cfg, seed=7):
    validate_training_data(data_tr, L, bs)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    t0 = time.time()
    model.train()
    for s, (x, y) in enumerate(batches(data_tr, bs, L, steps, gen), 1):
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab), y.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
        if s % max(1, steps // 5) == 0 or s == 1:
            print("    [%s] 步 %4d/%d loss %.4f ppl %.1f %.1fs" % (tag, s, steps, loss.item(),
                  __import__("math").exp(min(loss.item(), 20)), time.time() - t0))
            sys.stdout.flush()
    model.eval()


def save_checkpoint(path, model, cfg, train_args, corpus_meta, eval_tokens, overwrite=False,
                    tokenizer=None, extra=None):
    if not isinstance(eval_tokens, torch.Tensor) or eval_tokens.dtype != torch.long or eval_tokens.ndim not in (1, 2):
        raise ValueError("checkpoint eval_tokens 必须是 int64 一维流或二维 padded 样本")
    if not eval_tokens.numel() or int(eval_tokens.min()) < 0 or int(eval_tokens.max()) >= cfg.vocab:
        raise ValueError("checkpoint eval_tokens 不能为空且必须在 vocab 范围内")
    if tokenizer is not None and not isinstance(tokenizer, dict):
        raise ValueError("checkpoint tokenizer 必须是字典或 None")
    if extra is not None and not isinstance(extra, dict):
        raise ValueError("checkpoint extra 必须是字典")
    cfg_meta = dict(cfg.__dict__)
    expected_step = max(1, -(-int(train_args["seq"]) // int(cfg.n_buckets)))
    if cfg_meta.get("bucket_step", 0) <= 0:
        cfg_meta["bucket_step"] = expected_step
    elif int(cfg_meta["bucket_step"]) != expected_step:
        raise ValueError("model bucket_step does not match checkpoint training sequence length")
    if os.path.isdir(path):
        raise IsADirectoryError("checkpoint 路径是目录，拒绝覆盖: %s" % path)
    if os.path.exists(path) and not overwrite:
        raise FileExistsError("拒绝覆盖已有 checkpoint: %s (需显式 --overwrite_checkpoint)" % path)
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    payload = {
        "format": "engram-rule-checkpoint",
        "format_version": 1,
        "architecture_version": 1,
        "model": "EngramRuleLM",
        "cfg": cfg_meta,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "training": dict(train_args),
        "corpus": dict(corpus_meta),
        "eval_tokens": eval_tokens.detach().cpu().contiguous(),
        "tokenizer": tokenizer,
        "extra": dict(extra or {}),
    }
    fd, tmp = tempfile.mkstemp(prefix=".checkpoint-", suffix=".tmp", dir=parent)
    os.close(fd)
    try:
        torch.save(payload, tmp)
        if overwrite:
            os.replace(tmp, path)
        else:
            try:
                os.link(tmp, path)
            except FileExistsError as exc:
                raise FileExistsError("拒绝覆盖已有 checkpoint: %s" % path) from exc
            os.unlink(tmp)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_checkpoint(path, device):
    if not os.path.isfile(path):
        raise FileNotFoundError("checkpoint 文件不存在: %s" % path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError("当前 PyTorch 不支持 weights_only 安全加载；请升级 PyTorch 后重试") from exc
    if not isinstance(payload, dict):
        raise ValueError("checkpoint 根对象必须是字典")
    if payload.get("format") != "engram-rule-checkpoint" or payload.get("format_version") != 1:
        raise ValueError("不支持的 checkpoint 格式或版本")
    if payload.get("architecture_version") != 1:
        raise ValueError("不支持的 EngramRuleLM 架构版本")
    if payload.get("model") != "EngramRuleLM":
        raise ValueError("checkpoint 不是 EngramRuleLM")
    if payload.get("tokenizer") is not None and not isinstance(payload.get("tokenizer"), dict):
        raise ValueError("checkpoint tokenizer 元数据格式错误")
    required_training = {"steps", "bs", "seq", "lr", "train_seed", "eval_seed"}
    if not isinstance(payload.get("training"), dict) or not required_training <= payload["training"].keys():
        raise ValueError("checkpoint training 元数据缺字段")
    seq = int(payload["training"]["seq"])
    expected_step = max(1, -(-seq // int(payload["cfg"]["n_buckets"])))
    cfg_payload = dict(payload["cfg"])
    saved_step = int(cfg_payload.get("bucket_step", 0))
    if saved_step <= 0:
        cfg_payload["bucket_step"] = expected_step  # migrate pre-bucket-step checkpoints by training length
    elif saved_step != expected_step:
        raise ValueError("checkpoint bucket_step does not match training sequence length")
    if not isinstance(payload.get("eval_tokens"), torch.Tensor) or payload["eval_tokens"].dtype != torch.long:
        raise ValueError("checkpoint 缺少有效的整型评测 token")
    if payload["eval_tokens"].ndim not in (1, 2) or not payload["eval_tokens"].numel():
        raise ValueError("checkpoint 评测 token 必须为非空一维或二维张量")
    if int(payload["eval_tokens"].min()) < 0 or int(payload["eval_tokens"].max()) >= int(payload["cfg"]["vocab"]):
        raise ValueError("checkpoint 评测 token 超出模型词表范围")
    if not isinstance(payload.get("corpus"), dict) or "kind" not in payload["corpus"]:
        raise ValueError("checkpoint corpus 元数据缺字段")
    if payload.get("tokenizer") is not None:
        tok = payload["tokenizer"]
        if tok.get("kind") == "character":
            size = len(tok.get("id_to_token", []))
        elif tok.get("kind") == "bpe":
            size = int(tok.get("vocab_size", -1))
            if not isinstance(tok.get("json"), str) or not tok["json"]:
                raise ValueError("BPE checkpoint 缺少 tokenizer JSON")
        else:
            raise ValueError("不支持的 checkpoint tokenizer 类型")
        if size != int(payload["cfg"]["vocab"]) or payload["eval_tokens"].ndim != 2:
            raise ValueError("自然语言 checkpoint tokenizer 词表或 padded eval 样本不匹配")
    else:
        required_corpus = {"kind", "vocab_mapping", "tokenizer", "tokens", "alpha", "seed",
                           "train_fraction", "engram_build_tokens"}
        if not required_corpus <= payload["corpus"].keys() or payload["eval_tokens"].ndim != 1:
            raise ValueError("合成 checkpoint 的语料元数据或 token 流维度不正确")
    cfg = RCfg(**cfg_payload)
    model = EngramRuleLM(cfg)
    model.load_state_dict(payload["state_dict"], strict=True)
    del payload["state_dict"]
    model.to(device).eval()
    return model, cfg, payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--save_checkpoint")
    ap.add_argument("--load_checkpoint")
    ap.add_argument("--overwrite_checkpoint", action="store_true")
    ap.add_argument("--skip_dense", action="store_true",
                    help="只训练/评测 Engram-Rule，不额外训练稠密基线")
    a = ap.parse_args()
    if a.load_checkpoint and a.save_checkpoint:
        ap.error("--load_checkpoint 与 --save_checkpoint 不能同时使用")
    if a.load_checkpoint and a.overwrite_checkpoint:
        ap.error("--overwrite_checkpoint 不能与 --load_checkpoint 同时使用")
    if a.load_checkpoint and a.skip_dense:
        ap.error("加载 checkpoint 时稠密基线本来就不会训练，无需 --skip_dense")
    if a.overwrite_checkpoint and not a.save_checkpoint:
        ap.error("--overwrite_checkpoint 只对 --save_checkpoint 有效")

    corpus_meta = dict(kind="synthetic_zipf", vocab_mapping=None, tokenizer=None,
                       tokens=400_000, alpha=1.1, seed=0, train_fraction=0.8,
                       engram_build_tokens=200_000)
    payload = None
    dtr = dte = None
    eval_seed = EVAL_SEED
    if a.load_checkpoint:
        er, cfg, payload = load_checkpoint(a.load_checkpoint, DEV)
        corpus_meta = payload["corpus"]
        if corpus_meta.get("kind") != "synthetic_zipf" or corpus_meta.get("vocab_mapping") is not None:
            raise ValueError("本训练脚本只支持其已知的 Zipf 合成 token ID 评测口径")
        seq = int(payload["training"]["seq"])
        bs = int(payload["training"]["bs"])
        validate_eval_data(payload["eval_tokens"], seq)
        train_seed = int(payload["training"]["train_seed"])
        eval_seed = int(payload["training"]["eval_seed"])
        dte = payload["eval_tokens"]
        del payload["eval_tokens"]
        corpus_meta["eval_seed"] = eval_seed
        print("已加载 checkpoint %s (训练与语料重建均跳过)" % a.load_checkpoint)
    else:
        if a.save_checkpoint and os.path.exists(a.save_checkpoint) and not a.overwrite_checkpoint:
            raise FileExistsError("拒绝覆盖已有 checkpoint: %s" % a.save_checkpoint)
        if a.steps < 1 or a.bs < 1 or a.seq < 2 or a.lr <= 0:
            ap.error("steps/bs/lr 必须为正，seq 至少为 2")
        cfg = RCfg(vocab=4096, d=128, n_layers=4, n_heads=4, ffn_mult=2, rope_dim=16,
                   ngram_max=3, n_hash=3, n_slots=1 << 17, min_count=3, n_buckets=16,
                   bucket_step=max(1, -(-a.seq // 16)))
        data = zipf_data(cfg, corpus_meta["tokens"], seed=corpus_meta["seed"], alpha=corpus_meta["alpha"])
        ntr = int(len(data) * corpus_meta["train_fraction"])
        dtr, dte = data[:ntr], data[ntr:].clone()
        del data
        validate_training_data(dtr, a.seq, a.bs)
        er = EngramRuleLM(cfg).to(DEV)
        info = er.engram.build_from_corpus(dtr.view(1, -1)[:, :corpus_meta["engram_build_tokens"]])
        print("Engram 语料初始化: %s" % info)
        train_model(er, dtr, a.seq, a.steps, a.bs, a.lr, "ER", cfg)
        if a.save_checkpoint:
            save_checkpoint(a.save_checkpoint, er, cfg,
                            dict(steps=a.steps, bs=a.bs, seq=a.seq, lr=a.lr,
                                 train_seed=TRAIN_SEED, eval_seed=EVAL_SEED), corpus_meta, dte,
                            overwrite=a.overwrite_checkpoint)
            print("已保存 checkpoint %s" % os.path.abspath(a.save_checkpoint))
        seq, bs = a.seq, a.bs
        validate_eval_data(dte, seq)

    print("设备 %s | d=%d 层=%d | 词表 %d | Engram 槽 %d | 数据=%s (无自然语言 tokenizer)"
          % (DEV, cfg.d, cfg.n_layers, cfg.vocab, cfg.n_slots, corpus_meta["kind"]))
    r_er = evaluate(er, dte, seq, bs, cfg, seed=eval_seed)
    r_dn = None
    if not a.load_checkpoint and not a.skip_dense:
        dn = DenseLM(cfg).to(DEV)
        train_model(dn, dtr, seq, a.steps, a.bs, a.lr, "DN", cfg)
        r_dn = evaluate(dn, dte, seq, a.bs, cfg)

    st = er.stats(); tri = er.engram.export_rules(5)
    print("\n================ 指标台 (Zipf 合成 token，不是对话评测) ================")
    print("  Engram-Rule   ppl %.2f | 下一词准确率 %.4f | tokens/s %.0f"
          % (r_er["ppl"], r_er["acc"], r_er["tok_s"]))
    if r_dn is not None:
        print("  稠密基线      ppl %.2f | 下一词准确率 %.4f | tokens/s %.0f | ER/稠密 %.2f×"
              % (r_dn["ppl"], r_dn["acc"], r_dn["tok_s"], r_er["tok_s"] / r_dn["tok_s"]))
    print("  参数量        Engram-Rule %.3fM (Engram 表 %.3fM)"
          % (r_er["params"] / 1e6, sum(p.numel() for p in er.engram.parameters()) / 1e6))
    print("  稀疏度        存活槽 %d/%d = %.2f%%; 规则路径跳过率 %.3f"
          % (st["alive_slots"], st["n_slots"], 100 * st["alive_slots"] / st["n_slots"],
             st["rule_skip_ratio"]))
    print("  加速比上界    规则路径每位置 %d 次点积 vs 序列长度 %d ⇒ %.1f×"
          % (cfg.n_cand, seq, seq / cfg.n_cand))
    print("  可解释性      权重最高的规则槽 (前 5): %s" % tri)
    print("  数据边界      vocab ID 是合成编号; checkpoint 不含 tokenizer/自然语言词表")
    return 0


if __name__ == "__main__":
    sys.exit(main())
