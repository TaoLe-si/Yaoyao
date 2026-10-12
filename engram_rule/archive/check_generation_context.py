"""Compare next-answer logits for teacher-forced padded vs free-generation contexts.

This is an inference-only diagnostic; it does not update model weights.
"""
import argparse
import copy
import json
import torch

from dialogue_small import DEV, encode_examples, load_dialogue_checkpoint, format_prefix

CHECKPOINT = "engram_rule_dialogue_memorize8_v1.pt"


@torch.no_grad()
def query_state(model, tokens, query_pos, length):
    x = tokens.to(DEV).view(1, -1)
    for block in model.blocks:
        block.attn.record = True
    logits = model(x)[0, query_pos].float().cpu()
    starts, offsets = [], []
    for block in model.blocks:
        attn = block.attn
        starts.append(int(attn.last_startb[0, query_pos]))
        offsets.append(int(attn.last_window_off[0, query_pos, 0]))
    return logits, starts, offsets, length


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assert", dest="assert_parity", action="store_true")
    a = ap.parse_args()
    model, cfg, tok, _, _, _, payload = load_dialogue_checkpoint(CHECKPOINT, DEV)
    model.eval()
    legacy_cfg = copy.deepcopy(cfg)
    legacy_cfg.bucket_step = 0
    from model import EngramRuleLM
    legacy_model = EngramRuleLM(legacy_cfg).to(DEV)
    legacy_model.load_state_dict(model.state_dict(), strict=True)
    legacy_model.eval()
    train_seq = int(payload["training"]["seq"])
    nb = int(cfg.n_buckets)
    train_step = max(1, (train_seq + nb - 1) // nb)
    prompts = payload["extra"]["experiment"]["prompts"]
    rows = prompts["train"] + prompts["eval"]
    out = []
    for row in rows:
        full, starts, lengths = encode_examples([row], tok, train_seq)
        answer_start = int(starts[0])
        prefix_ids = tok.encode(format_prefix(row), add_special_tokens=False).ids
        if prefix_ids != full[0, :answer_start].tolist():
            raise AssertionError("teacher-forced row prefix differs from generation prompt")
        query_pos = answer_start - 1
        logit_train, start_train, off_train, _ = query_state(model, full[0], query_pos, train_seq)
        short = torch.tensor(prefix_ids, dtype=torch.long)
        logit_gen, start_gen, off_gen, gen_len = query_state(model, short, query_pos, len(prefix_ids))
        logit_legacy, _, _, _ = query_state(legacy_model, full[0], query_pos, train_seq)
        logit_legacy_gen, _, _, _ = query_state(legacy_model, short, query_pos, len(prefix_ids))
        diff = float((logit_train - logit_gen).abs().max())
        legacy_diff = float((logit_legacy - logit_legacy_gen).abs().max())
        cos = float(torch.nn.functional.cosine_similarity(
            logit_train.view(1, -1), logit_gen.view(1, -1)).item())
        out.append(dict(question=row["question"], prompt_tokens=len(prefix_ids),
                        train_seq=train_seq, fixed_bucket_step=int(cfg.bucket_step),
                        legacy_prompt_bucket_step=max(1, (gen_len + nb - 1) // nb),
                        max_abs_logit_diff=diff, legacy_dynamic_max_abs_diff=legacy_diff,
                        logit_cosine=cos,
                        top1_train=int(logit_train.argmax()), top1_generation=int(logit_gen.argmax()),
                        top1_equal=bool(logit_train.argmax() == logit_gen.argmax()),
                        start_train=start_train, start_generation=start_gen,
                        offset_train=off_train, offset_generation=off_gen))
    fixed_max = max(x["max_abs_logit_diff"] for x in out)
    legacy_max = max(x["legacy_dynamic_max_abs_diff"] for x in out)
    top1_agreement = sum(x["top1_equal"] for x in out) / len(out)
    summary = dict(checkpoint=CHECKPOINT, seed=payload["training"].get("train_seed"),
                   examples=len(out), max_abs_logit_diff=fixed_max,
                   legacy_dynamic_max_abs_diff=legacy_max,
                   mean_logit_cosine=sum(x["logit_cosine"] for x in out) / len(out),
                   top1_agreement=top1_agreement,
                   fixed_bucket_steps=sorted({x["fixed_bucket_step"] for x in out}),
                   legacy_prompt_bucket_steps=sorted({x["legacy_prompt_bucket_step"] for x in out}),
                   examples_detail=out)
    print(json.dumps(summary, ensure_ascii=False))
    if a.assert_parity:
        assert fixed_max < 1e-4 and top1_agreement == 1.0, "fixed bucket_step padded/unpadded logits diverged"
        assert legacy_max > fixed_max * 100, "legacy dynamic bucket_step negative control did not fail strongly"
        print("PASS fixed bucket_step parity; legacy dynamic-step negative control fails")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
