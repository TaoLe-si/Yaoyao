"""Compare routing resolutions on one fixed checkpoint without training."""
import json

import torch

from dialogue_small import DEV, answer_metrics, evaluate_generations, format_prefix, load_dialogue_checkpoint
from model import EngramRuleLM, RCfg

CHECKPOINT = "engram_rule_dialogue_bpe_answer_disjoint_smoke_v7.pt"
STEPS = (32, 16, 8, 6)


def main():
    model, cfg, tok, eval_x, eval_starts, eval_lengths, payload = \
        load_dialogue_checkpoint(CHECKPOINT, DEV)
    experiment = payload.get("extra", {}).get("experiment", {})
    rows = experiment.get("prompts", {}).get("eval", [])
    generation = experiment.get("generation", {})
    if len(rows) != 12:
        raise ValueError("diagnostic requires the fixed 12 heldout prompts")
    results = []
    for step in STEPS:
        if step == cfg.bucket_step:
            candidate = model
        else:
            values = dict(cfg.__dict__)
            values["bucket_step"] = step
            candidate = EngramRuleLM(RCfg(**values)).to(DEV).eval()
            candidate.load_state_dict(model.state_dict(), strict=True)
        metric = answer_metrics(candidate, eval_x, eval_starts, eval_lengths, 64, DEV)
        generation_metric, outputs = evaluate_generations(
            candidate, tok, rows, device=DEV, **generation)
        first_targets, first_predictions, first_ranks = [], [], []
        candidate.eval()
        with torch.no_grad():
            for row in rows:
                prompt_ids = tok.encode(format_prefix(row), add_special_tokens=False).ids
                full_ids = tok.encode(format_prefix(row) + row["answer"] + "[EOS]",
                                      add_special_tokens=False).ids
                if full_ids[:len(prompt_ids)] != prompt_ids:
                    raise AssertionError("generation prompt is not a prefix of the target sequence")
                logits = candidate(torch.tensor([prompt_ids], dtype=torch.long, device=DEV))[0, -1].float()
                target = full_ids[len(prompt_ids)]
                first_targets.append(target)
                first_predictions.append(int(logits.argmax()))
                first_ranks.append(int((logits > logits[target]).sum()) + 1)
        results.append(dict(bucket_step=step, teacher_forced=metric,
                            first_answer_token_accuracy=sum(a == b for a, b in
                                                            zip(first_targets, first_predictions)) / len(rows),
                            first_answer_token_mean_rank=sum(first_ranks) / len(first_ranks),
                            generation=generation_metric,
                            generations=[dict(question=o["question"], reference=o["reference"],
                                             generated=o["generated"]) for o in outputs]))
        if candidate is not model:
            del candidate
        if DEV == "cuda":
            torch.cuda.empty_cache()
    print(json.dumps(dict(checkpoint=CHECKPOINT, weights_fixed=True, training_performed=False,
                          baseline_bucket_step=cfg.bucket_step, results=results),
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
