"""Eight-example free-generation memorization control using the saved v3 BPE/config."""
import json
import os
import torch

from dialogue_small import (DEV, answer_metrics, build_engram_from_examples,
                            evaluate_generations, generation_config_from_args,
                            load_dialogue_checkpoint, train_dialogue)
from model import EngramRuleLM
from train import RCfg

SOURCE_CHECKPOINT = "engram_rule_dialogue_bpe_learning_v3.pt"
OUTPUT_CHECKPOINT = "engram_rule_dialogue_memorize8_budget192_v3.pt"
OUTPUT_REPORT = "dialogue_qa_memorize8_budget192_v3.jsonl"
STEPS = 200
BATCH = 256
LR = 3e-3
RELOAD_REPORT = "dialogue_qa_memorize8_budget192_v3_reload.jsonl"
MAX_LEN = 192
MAX_NEW_TOKENS = MAX_LEN
GENERATION = dict(max_new_tokens=MAX_NEW_TOKENS, context_limit=MAX_LEN, temperature=0.0,
                  repetition_penalty=1.0, no_repeat_ngram=0, utf8_safe=True, seed=0)
SEEDS = (101, 202, 303)


def summarize(values):
    return dict(mean=sum(values) / len(values), min=min(values), max=max(values),
                range=max(values) - min(values))


def require_reference_budget(starts, lengths, generation, split):
    required = lengths - starts  # Complete answer, including [EOS].
    available = torch.minimum(torch.full_like(starts, generation["max_new_tokens"]),
                              generation["context_limit"] - starts)
    if bool((required > available).any()):
        raise ValueError("%s reference exceeds generation budget" % split)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--load_checkpoint")
    ap.add_argument("--max_new_tokens", type=int)
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--repetition_penalty", type=float)
    ap.add_argument("--no_repeat_ngram", type=int)
    ap.add_argument("--generation_seed", type=int)
    ap.add_argument("--reload_report", default=RELOAD_REPORT)
    a = ap.parse_args()
    if a.load_checkpoint:
        from dialogue_small import encode_examples
        model, cfg, tok, _, _, _, payload = load_dialogue_checkpoint(a.load_checkpoint, DEV)
        saved = payload.get("extra", {}).get("experiment", {}).get("generation", GENERATION)
        generation_args = argparse.Namespace(max_new_tokens=a.max_new_tokens,
                                             context_limit=None, temperature=a.temperature,
                                             repetition_penalty=a.repetition_penalty,
                                             no_repeat_ngram=a.no_repeat_ngram,
                                             generation_seed=a.generation_seed,
                                             unsafe_utf8=False)
        from dialogue_small import generation_config_from_args
        generation = generation_config_from_args(generation_args,
                                                 int(payload["training"]["seq"]), saved)
        prompts = payload["extra"]["experiment"]["prompts"]
        context_limit = int(payload["training"]["seq"])
        tx, ts, tl = encode_examples(prompts["train"], tok, context_limit)
        ex, es, el = encode_examples(prompts["eval"], tok, context_limit)
        tmetric = answer_metrics(model, tx, ts, tl, BATCH, DEV)
        emetric = answer_metrics(model, ex, es, el, BATCH, DEV)
        tg, tout = evaluate_generations(model, tok, prompts["train"], device=DEV, **generation)
        eg, eout = evaluate_generations(model, tok, prompts["eval"], device=DEV, **generation)
        report = dict(type="memorize8_reload", train_teacher_forced=tmetric,
                      eval_teacher_forced=emetric, train_generation=tg, eval_generation=eg,
                      generation=generation, checkpoint=a.load_checkpoint)
        if os.path.exists(a.reload_report):
            raise FileExistsError("refusing to overwrite reload report")
        with open(a.reload_report, "x", encoding="utf-8") as f:
            f.write(json.dumps(report, ensure_ascii=False) + "\n")
            for split, items in (("train", tout), ("eval", eout)):
                for item in items:
                    f.write(json.dumps(dict(type="generation", split=split, **item),
                                       ensure_ascii=False) + "\n")
        print(json.dumps(report, ensure_ascii=False))
        print("saved reload report", os.path.abspath(a.reload_report))
        return 0
    if os.path.exists(OUTPUT_CHECKPOINT) or os.path.exists(OUTPUT_REPORT):
        raise FileExistsError("refusing to overwrite memorization-control outputs")
    _, _, tok, _, _, _, payload = load_dialogue_checkpoint(SOURCE_CHECKPOINT, "cpu")
    prompts = payload["extra"]["experiment"]["prompts"]
    train_rows = prompts["train"]
    eval_rows = prompts["eval"]
    if len(train_rows) != 8 or len(eval_rows) != 12:
        raise ValueError("expected fixed source prompts: 8 train, 12 heldout")
    norm = lambda s: "".join(s.casefold().split())
    assert {norm(r["question"]) for r in train_rows}.isdisjoint(norm(r["question"]) for r in eval_rows)
    assert {norm(r["answer"]) for r in train_rows}.isdisjoint(norm(r["answer"]) for r in eval_rows)

    from dialogue_small import encode_examples
    train_x, train_starts, train_lengths = encode_examples(train_rows, tok, MAX_LEN)
    eval_x, eval_starts, eval_lengths = encode_examples(eval_rows, tok, MAX_LEN)
    if train_x.shape[0] != 8 or eval_x.shape[0] != 12:
        raise ValueError("memorization control requires all 8/12 QA rows within the BPE context limit")
    assert max(int(train_lengths.max()), int(eval_lengths.max())) <= MAX_LEN
    require_reference_budget(train_starts, train_lengths, GENERATION, "train")
    require_reference_budget(eval_starts, eval_lengths, GENERATION, "eval")
    cfg_values = dict(payload["cfg"])
    cfg_values["pos_abs_max"] = max(int(cfg_values["pos_abs_max"]), MAX_LEN)
    source_bucket_step = int(cfg_values.get("bucket_step", 0))
    target_bucket_step = max(1, -(-MAX_LEN // int(cfg_values["n_buckets"])))
    cfg_values["bucket_step"] = target_bucket_step
    cfg = RCfg(**cfg_values)
    results = []
    generation_details = []
    seed0_state = None
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = EngramRuleLM(cfg).to(DEV)
        alive = build_engram_from_examples(model, train_x, train_lengths)
        before_train = answer_metrics(model, train_x, train_starts, train_lengths, BATCH, DEV)
        before_eval = answer_metrics(model, eval_x, eval_starts, eval_lengths, BATCH, DEV)
        before_train_gen, _ = evaluate_generations(model, tok, train_rows, device=DEV, **GENERATION)
        before_eval_gen, _ = evaluate_generations(model, tok, eval_rows, device=DEV, **GENERATION)
        train_dialogue(model, train_x, train_starts, train_lengths, STEPS, BATCH, LR, DEV, seed)
        after_train = answer_metrics(model, train_x, train_starts, train_lengths, BATCH, DEV)
        after_eval = answer_metrics(model, eval_x, eval_starts, eval_lengths, BATCH, DEV)
        after_train_gen, train_out = evaluate_generations(model, tok, train_rows, device=DEV, **GENERATION)
        after_eval_gen, eval_out = evaluate_generations(model, tok, eval_rows, device=DEV, **GENERATION)
        results.append(dict(seed=seed, alive=alive,
                            before_train=before_train, after_train=after_train,
                            before_eval=before_eval, after_eval=after_eval,
                            before_train_generation=before_train_gen,
                            after_train_generation=after_train_gen,
                            before_eval_generation=before_eval_gen,
                            after_eval_generation=after_eval_gen))
        generation_details.append(dict(seed=seed, train=train_out, eval=eval_out))
        print("seed %d: train NLL %.3f→%.3f, EM %.3f→%.3f; heldout NLL %.3f→%.3f, EM %.3f charF1 %.3f" %
              (seed, before_train["answer_nll"], after_train["answer_nll"],
               before_train_gen["exact_match"], after_train_gen["exact_match"],
               before_eval["answer_nll"], after_eval["answer_nll"],
               after_eval_gen["exact_match"], after_eval_gen["char_f1"]))
        if seed == SEEDS[0]:
            seed0_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        del model
        if DEV == "cuda":
            torch.cuda.empty_cache()

    summary = dict(task="8-example free-generation memorization control",
                   steps=STEPS, batch=BATCH, lr=LR, seeds=list(SEEDS),
                   context_limit=MAX_LEN, source_bucket_step=source_bucket_step,
                   fixed_bucket_step=int(cfg.bucket_step),
                   train_nll_before=summarize([r["before_train"]["answer_nll"] for r in results]),
                   train_nll_after=summarize([r["after_train"]["answer_nll"] for r in results]),
                   train_em_before=summarize([r["before_train_generation"]["exact_match"] for r in results]),
                   train_em_after=summarize([r["after_train_generation"]["exact_match"] for r in results]),
                   heldout_nll_before=summarize([r["before_eval"]["answer_nll"] for r in results]),
                   heldout_nll_after=summarize([r["after_eval"]["answer_nll"] for r in results]),
                   heldout_em_after=summarize([r["after_eval_generation"]["exact_match"] for r in results]),
                   heldout_char_f1_after=summarize([r["after_eval_generation"]["char_f1"] for r in results]),
                   seeds_memorizing_at_least_7_of_8=sum(
                       r["after_train_generation"]["exact_match"] >= 7 / 8 for r in results),
                   criterion_train_em_at_least_0_875="memorizes 7/8 training answers",
                   generation_limit=MAX_NEW_TOKENS, individual=results)
    with open(OUTPUT_REPORT, "x", encoding="utf-8") as f:
        f.write(json.dumps(dict(type="summary", data=summary), ensure_ascii=False) + "\n")
        for item in generation_details:
            for split in ("train", "eval"):
                for row in item[split]:
                    f.write(json.dumps(dict(type="generation", seed=item["seed"], split=split, **row),
                                       ensure_ascii=False) + "\n")

    from dialogue_small import save_dialogue_checkpoint
    model = EngramRuleLM(cfg)
    model.load_state_dict(seed0_state, strict=True)
    model.to(DEV).eval()
    metadata = dict(payload["corpus"])
    metadata["kind"] = "synthetic-source-eight-example-overfit-control"
    metadata["scope"] = "only eight source train QA rows were used to fit weights; heldout rows are not included in train gradients"
    # Store an evaluation matrix aligned to this control's longer 192-token context.
    eval_tokens, starts, lengths = __import__("dialogue_small").encode_examples(eval_rows, tok, MAX_LEN)
    save_dialogue_checkpoint(OUTPUT_CHECKPOINT, model, cfg, tok,
                             dict(steps=STEPS, bs=BATCH, seq=MAX_LEN, lr=LR,
                                  train_seed=SEEDS[0], eval_seed=0, seeds=list(SEEDS)),
                             metadata, eval_tokens, starts, lengths,
                             generation_rows=eval_rows,
                             experiment=dict(prompts=dict(train=train_rows, eval=eval_rows),
                                             summary=summary, generation=GENERATION,
                                             generation_limit=MAX_NEW_TOKENS),
                             source_rows=train_rows + eval_rows,
                             source_rows_complete=False)
    print("summary", json.dumps(summary, ensure_ascii=False))
    print("saved checkpoint", os.path.abspath(OUTPUT_CHECKPOINT))
    print("saved generation report", os.path.abspath(OUTPUT_REPORT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
