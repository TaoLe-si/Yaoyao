"""Inspect first-answer predictions and question dependence in saved dialogue models."""
import argparse
import json

import torch

from dialogue_small import (bytelevel_token_bytes, format_prefix, format_record,
                            load_dialogue_checkpoint, select_generation_rows,
                            split_records, _select_next_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    model, cfg, tok, _, _, _, payload = load_dialogue_checkpoint(args.checkpoint, "cpu")
    prompts = payload.get("extra", {}).get("experiment", {}).get("prompts", {})
    if "train" in prompts and "eval" in prompts:
        selected = {key: prompts[key] for key in ("train", "eval")}
    else:
        rows = payload["extra"]["raw_dialogue_rows"]
        train, eval_rows = split_records(rows, seed=29)
        selected_train, selected_eval = select_generation_rows(train, eval_rows)
        selected = dict(train=selected_train, eval=selected_eval)
    byte_map, special_ids = bytelevel_token_bytes(tok)
    eos_id = tok.token_to_id("[EOS]")
    unk_id = tok.token_to_id("[UNK]")
    results = {}
    model.eval()
    for split, rows in selected.items():
        items = []
        for row in rows:
            prompt_ids = tok.encode(format_prefix(row), add_special_tokens=False).ids
            full_ids = tok.encode(format_record(row), add_special_tokens=False).ids
            answer_start = len(prompt_ids)
            if prompt_ids != full_ids[:answer_start] or answer_start >= len(full_ids):
                raise AssertionError("generation prompt differs from the teacher-forced prefix")
            for block in model.blocks:
                block.attn.record = True
            with torch.no_grad():
                logits = model(torch.tensor([prompt_ids]))[0, -1].float()
                offsets = [int(block.attn.last_window_off[0, -1, 0]) for block in model.blocks]
                buckets = [int(block.attn.last_startb[0, -1]) for block in model.blocks]
                changed_ids = prompt_ids.copy()
                changed_ids[2:-1] = [unk_id] * max(0, len(changed_ids) - 3)
                changed_logits = model(torch.tensor([changed_ids]))[0, -1].float()
            target = int(full_ids[answer_start])
            safe_id, _ = _select_next_id(logits, [], (0, 0x80, 0xBF), byte_map,
                                         special_ids, eos_id, 0.0, 1.0, 0)
            items.append(dict(question=row["question"], gold_first_id=target,
                              raw_first_id=int(logits.argmax()), safe_first_id=safe_id,
                              gold_rank=int((logits > logits[target]).sum()) + 1,
                              changed_top1=int(changed_logits.argmax()),
                              offsets=offsets, buckets=buckets))
        results[split] = dict(examples=len(items), raw_first_accuracy=sum(
            item["raw_first_id"] == item["gold_first_id"] for item in items) / len(items),
                              safe_first_accuracy=sum(
            item["safe_first_id"] == item["gold_first_id"] for item in items) / len(items),
                              question_ablation_flips=sum(
            item["raw_first_id"] != item["changed_top1"] for item in items),
                              items=items)
    print(json.dumps(dict(checkpoint=args.checkpoint, bucket_step=cfg.bucket_step,
                          sequence_length=payload["training"]["seq"], splits=results),
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
