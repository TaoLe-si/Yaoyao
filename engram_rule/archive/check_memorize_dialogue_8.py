"""Offline guards for the eight-example memorization control."""
import os
import tempfile
import torch

from dialogue_small import (build_bpe, encode_examples, evaluate_generations,
                            format_record, format_prefix, greedy_generate,
                            load_dialogue_checkpoint, normalize_text, parse_record,
                            split_records)
from memorize_dialogue_8 import require_reference_budget
from model import EngramRuleLM, RCfg


def main():
    rows = []
    for i in range(24):
        rows.append(dict(question="独特问题%02d的内容是什么？" % i,
                         answer="独特答案%02d包含天文知识。" % i,
                         history=[]))
    # Explicit repeated-question/repeated-answer links must form one component.
    rows.extend([
        dict(question=" 独特问题00 的内容是什么？", answer="另一种问法的答案。", history=[]),
        dict(question="完全不同的问题？", answer="独特答案01包含天文知识。", history=[]),
    ])
    train, held, meta = split_records(rows, train_fraction=0.75, seed=12,
                                     min_groups=20, return_meta=True)
    tq = {normalize_text(r["question"]) for r in train}
    eq = {normalize_text(r["question"]) for r in held}
    ta = {normalize_text(r["answer"]) for r in train}
    ea = {normalize_text(r["answer"]) for r in held}
    assert not tq & eq and not ta & ea
    assert meta["question_overlap"] == meta["answer_overlap"] == 0
    assert train and held and len(train) + len(held) == len(rows)

    tok = build_bpe([format_record(r) for r in train], vocab_size=300)
    tr_x, tr_s, tr_l, tr_kept = encode_examples(train, tok, 96, return_rows=True)
    ev_x, ev_s, ev_l, ev_kept = encode_examples(held, tok, 96, return_rows=True)
    assert tr_x.shape[0] == len(train) == len(tr_kept)
    assert ev_x.shape[0] == len(held) == len(ev_kept)
    require_reference_budget(tr_s, tr_l, dict(max_new_tokens=96, context_limit=96), "train")
    short_budget = int((tr_l - tr_s).max()) - 1
    assert short_budget > 0
    try:
        require_reference_budget(tr_s, tr_l,
                                 dict(max_new_tokens=short_budget, context_limit=96), "train")
    except ValueError:
        pass
    else:
        raise AssertionError("negative control: a short cap must reject an unfinishable reference")
    cfg = RCfg(vocab=tok.get_vocab_size(), d=32, n_layers=1, n_heads=4,
               ffn_mult=2, rope_dim=8, ngram_max=2, n_hash=2, n_slots=512,
               min_count=1, n_buckets=8, gate_rank=8, pos_abs_max=128,
               window_max=8)
    model = EngramRuleLM(cfg).eval()
    x0 = tok.encode(format_prefix(tr_kept[0]), add_special_tokens=False).ids
    seen = []
    class Capture(torch.nn.Module):
        def forward(self, x):
            seen.append(x.detach().cpu().flatten().tolist())
            return torch.nn.functional.one_hot(x, cfg.vocab).float()
    capture = Capture()
    result = greedy_generate(capture, tok, tr_kept[0], max_new_tokens=2, device="cpu")
    assert seen[0] == x0
    assert result["reference"] not in "".join(tok.id_to_token(i) or "" for i in seen[0])

    # A real random model exercises fixed heldout and training-sized prompt metrics.
    eval_gen = ev_kept
    metrics, outputs = evaluate_generations(model, tok, eval_gen, 4, "cpu")
    assert metrics["examples"] == len(outputs) == len(eval_gen)
    assert all(o["generated_tokens"] <= 4 for o in outputs)
    assert 0.0 <= metrics["exact_match"] <= 1.0 and 0.0 <= metrics["char_f1"] <= 1.0

    print("PASS answer/question connected groups, prompt-only greedy, EOS/length cap, bounded generation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
