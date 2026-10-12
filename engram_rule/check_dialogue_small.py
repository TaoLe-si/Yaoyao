"""Offline guards for the bounded BPE dialogue experiment.

Run: python engram_rule/check_dialogue_small.py
"""
import copy
import os
import tempfile

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model import EngramRuleLM, RCfg
from dialogue_small import (DATA_LICENSE, _select_next_id, answer_loss_sums, answer_metrics,
                            audit_text_rows,
                            build_bpe, build_engram_from_examples,
                            build_split_bpe, bytelevel_token_bytes, char_f1,
                            partition_by_token_budget, parse_record,
                            generation_config_from_args, encode_examples,
                            evaluate_dialogue, format_prefix, format_record, greedy_generate,
                            normalize_text, question_answer_conflicts,
                            sample_ranges, split_records, SPECIAL, utf8_state_for_bytes,
                            valid_token_bytes)
from train import load_checkpoint
from dialogue_small import save_dialogue_checkpoint


def main():
    raw = [
        '{"question":"火星离地球多远？","answer":"距离会随轨道位置变化。","history":[]}',
        '{"question":"火星有什么颜色？","answer":"火星表面常呈红色。","history":[]}',
        '{"question":"能再解释一下吗？","answer":"因为表面有富含氧化铁的尘埃。","history":[["火星为什么是红色？","主要因为表土中含有氧化铁。"]]}',
        '{"question":"给我一句总结。","answer":"铁氧化物尘埃让火星显红。","history":[["火星是什么颜色？","常见颜色是红色。"]]}',
        '{"question":"太阳会熄灭吗？","answer":"太阳最终会进入演化晚期。","history":[]}',
        '{"question":"那之后呢？","answer":"它会抛出外层并留下白矮星。","history":[["太阳会一直稳定吗？","不会，恒星会随时间演化。"]]}',
        '{"question":"什么是光年？","answer":"光年是光在一年中传播的距离。","history":[]}',
        '{"question":"用简单的话说呢？","answer":"它是衡量宇宙距离的单位。","history":[["光年怎么定义？","按光一年走过的距离定义。"]]}',
    ]
    rows = [parse_record(x) for x in raw]
    assert all(rows)
    assert normalize_text("火星 离地球多远？") == normalize_text("火星离地球多远？")
    train_rows, heldout = split_records(rows, train_fraction=0.625, seed=4)
    assert {r["answer"] for r in train_rows}.isdisjoint({r["answer"] for r in heldout})
    assert any(r["history"] for r in rows), "fixture must include history records"
    corrupted = [dict(rows[0], answer="答案��错误")]
    clean_audit = audit_text_rows(rows)
    corrupt_audit = audit_text_rows(corrupted)
    assert clean_audit["clean"] and not corrupt_audit["clean"]
    assert corrupt_audit["issue_counts"]["replacement_character"] == 1
    mojibake_audit = audit_text_rows([dict(rows[0], answer="FranÃ§ais")])
    assert mojibake_audit["issue_counts"].get("mojibake_signature", 0) == 1
    assert parse_record('{"question":"问题","answer":"坏�文本"}') is not None
    ambiguous_rows = [dict(rows[0], question="同一个问题？", answer="答案甲"),
                      dict(rows[1], question=" 同一个问题？ ", answer="答案乙"),
                      dict(rows[2], question="独立问题？", answer="独立答案"),
                      dict(rows[3], question="另一个独立问题？", answer="另一个独立答案")]
    conflict_meta = question_answer_conflicts(ambiguous_rows)
    assert len(ambiguous_rows) == 4
    assert conflict_meta["conflicting_prompts"] == 1
    assert set(conflict_meta["examples"][0]["answers"]) == {"答案甲", "答案乙"}
    linked_rows = rows + [dict(rows[0], question="火星 离地球 多远？", answer="距离会逐年改变。")]
    conflict_split_train, conflict_split_eval = split_records(
        ambiguous_rows, train_fraction=0.67, seed=17)
    train_conflicts = question_answer_conflicts(conflict_split_train)
    eval_conflicts = question_answer_conflicts(conflict_split_eval)
    assert len(conflict_split_train) + len(conflict_split_eval) == len(ambiguous_rows)
    assert train_conflicts["conflicting_prompts"] + eval_conflicts["conflicting_prompts"] == 1
    tr2, te2, split_meta = split_records(linked_rows, train_fraction=0.625, seed=5, return_meta=True)
    original_in_train = any(r["question"] == rows[0]["question"] for r in tr2)
    rewritten_in_train = any(r["answer"] == "距离会逐年改变。" for r in tr2)
    assert original_in_train == rewritten_in_train, \
        "same normalized question with different answers must stay in one split"
    assert {r["question"].replace(" ", "") for r in tr2}.isdisjoint(
        {r["question"].replace(" ", "") for r in te2})
    assert {r["answer"] for r in tr2}.isdisjoint({r["answer"] for r in te2})
    assert split_meta["question_overlap"] == split_meta["answer_overlap"] == 0
    tok, bpe_audit = build_split_bpe(train_rows, heldout, 300)
    train_only_tok = build_bpe([__import__("dialogue_small").format_record(r)
                                for r in train_rows], 300)
    assert tok.to_str() == train_only_tok.to_str(), "evaluation rows must not affect BPE fitting"
    assert bpe_audit["fit_scope"] == "train_only"
    assert bpe_audit["fit_rows"] == len(train_rows) and bpe_audit["eval_rows"] == len(heldout)
    assert all(tok.token_to_id(s) is not None for s in SPECIAL)
    assert audit_text_rows(rows, tok)["clean"], "clean fixture must round-trip through ByteLevel BPE"
    assert not audit_text_rows(corrupted, tok)["clean"], "corrupt fixture must be rejected before training"
    with_corrupt_train = train_rows + corrupted
    try:
        build_bpe([__import__("dialogue_small").format_record(r) for r in with_corrupt_train], 300)
    except ValueError:
        pass
    else:
        raise AssertionError("BPE accepted a U+FFFD-bearing training row")
    token_budget_rows, token_budget_meta = partition_by_token_budget(
        train_rows + heldout, tok, max_tokens=32)
    assert token_budget_meta["rejected"] > 0
    assert token_budget_meta["rejected"] + len(token_budget_rows) == len(train_rows + heldout)
    assert all(len(tok.encode(__import__("dialogue_small").format_record(r), add_special_tokens=False).ids) <= 32
               for r in token_budget_rows)
    no_answer_loss_rows = []
    for candidate in train_rows:
        prefix_len = len(tok.encode(format_prefix(candidate), add_special_tokens=False).ids)
        full_len = len(tok.encode(__import__("dialogue_small").format_record(candidate), add_special_tokens=False).ids)
        if full_len > prefix_len + 1:
            no_answer_loss_rows.append((candidate, prefix_len, full_len))
    assert no_answer_loss_rows
    candidate, prefix_len, full_len = no_answer_loss_rows[0]
    try:
        encode_examples([candidate], tok, prefix_len, return_rows=True)
    except ValueError:
        pass
    else:
        raise AssertionError("a truncated row with no answer target must not enter training")
    short_row = {"question": "Q", "answer": "A", "history": []}
    mixed_rows, _ = partition_by_token_budget([short_row, rows[0]], tok, max_tokens=6)
    short_x, short_s, short_l, kept = encode_examples([short_row, rows[0]], tok, 6,
                                                       return_rows=True)
    assert len(mixed_rows) == 1 and mixed_rows[0] == short_row
    assert kept == [short_row] and short_x.shape[0] == 1
    assert int(short_s[0]) >= 1 and int(short_l[0]) <= 6
    eos_id = tok.token_to_id("[EOS]")

    byte_map, special_ids = bytelevel_token_bytes(tok)
    assert len(byte_map) == tok.get_vocab_size() and eos_id in special_ids
    for valid in (b"ASCII", "中".encode("utf-8"), "🙂".encode("utf-8"), b"\xf4\x8f\xbf\xbf"):
        assert utf8_state_for_bytes(valid) == (0, 0x80, 0xBF)
    for invalid in (b"\x80", b"\xc0\x80", b"\xe0\x80\x80", b"\xed\xa0\x80",
                    b"\xf0\x80\x80\x80", b"\xf4\x90\x80\x80", b"\xf5\x80\x80\x80"):
        assert utf8_state_for_bytes(invalid) is None
    assert utf8_state_for_bytes(b"\xe4") == (2, 0x80, 0xBF)
    assert valid_token_bytes(utf8_state_for_bytes(b"\xe4"), b"\x4e") is None
    token_for_byte = lambda byte: next(i for i, fragment in byte_map.items()
                                       if fragment == bytes([byte]))
    ascii_a, ascii_b, ascii_c = (token_for_byte(b) for b in (0x41, 0x42, 0x43))
    lead_e4, bad_continuation = token_for_byte(0xE4), token_for_byte(0x4E)

    class FixedNext(torch.nn.Module):
        def __init__(self, vocab, next_id):
            super().__init__()
            self.vocab = vocab
            self.next_id = next_id
            self.inputs = []

        def forward(self, x):
            self.inputs.append(x.detach().cpu().clone())
            out = torch.zeros((x.shape[0], x.shape[1], self.vocab), device=x.device)
            out[:, :, self.next_id] = 1
            return out

    eos_id = tok.token_to_id("[EOS]")
    fake = FixedNext(tok.get_vocab_size(), eos_id)
    generated = greedy_generate(fake, tok, heldout[0], max_new_tokens=4, device="cpu")
    expected_prompt = tok.encode(format_prefix(heldout[0]), add_special_tokens=False).ids
    assert fake.inputs[0].flatten().tolist() == expected_prompt
    assert generated["eos"] and generated["generated_tokens"] == 1

    fake_cap = FixedNext(tok.get_vocab_size(), ascii_a)
    cap_result = greedy_generate(fake_cap, tok, heldout[0], max_new_tokens=3, device="cpu")
    assert cap_result["length_cap"] and cap_result["generated_tokens"] == 3
    assert not cap_result["truncated_context"], \
        "negative control: an explicit token cap must not count as context exhaustion"
    assert cap_result["generated"] == "AAA" and cap_result["repeated_token_count"] == 2
    full_reference = tok.encode(format_record(heldout[0]), add_special_tokens=False).ids
    assert full_reference[:len(expected_prompt)] == expected_prompt
    assert cap_result["reference_tokens"] == len(full_reference) - len(expected_prompt)
    tight_limit = len(expected_prompt) + 1
    fixed_context = FixedNext(tok.get_vocab_size(), ascii_a)
    fixed_context_result = greedy_generate(fixed_context, tok, heldout[0], max_new_tokens=3,
                                           device="cpu", context_limit=tight_limit)
    assert fixed_context_result["generation_budget"] == 1
    assert fixed_context_result["length_cap"] and fixed_context_result["truncated_context"]
    assert fixed_context_result["generated_tokens"] == 1
    assert fixed_context.inputs[0].flatten().tolist() == expected_prompt
    # Negative control for the old sliding suffix: it would renumber these tokens at position 0.
    legacy_overflow = expected_prompt + [ascii_a, ascii_a]
    assert legacy_overflow[-tight_limit:] != legacy_overflow
    try:
        greedy_generate(FixedNext(tok.get_vocab_size(), ascii_a), tok, heldout[0],
                        max_new_tokens=1, device="cpu", context_limit=len(expected_prompt) - 1)
    except ValueError:
        pass
    else:
        raise AssertionError("overlong prompt must not be silently renumbered after truncation")
    capped_metrics, _ = __import__("dialogue_small").evaluate_generations(
        FixedNext(tok.get_vocab_size(), ascii_a), tok, [heldout[0]], 3, "cpu",
        context_limit=tight_limit)
    assert capped_metrics["reference_longer_than_budget"] == 1
    assert capped_metrics["truncated_context_rate"] == 1.0
    roomy_limit = len(full_reference) + 1
    roomy_metrics, _ = __import__("dialogue_small").evaluate_generations(
        FixedNext(tok.get_vocab_size(), eos_id), tok, [heldout[0]], roomy_limit, "cpu",
        context_limit=roomy_limit)
    assert roomy_metrics["reference_longer_than_budget"] == 0
    assert roomy_metrics["truncated_context_rate"] == 0.0
    cap_metrics, duplicate_outputs = __import__("dialogue_small").evaluate_generations(
        fake_cap, tok, [heldout[0], heldout[0]], max_new_tokens=2, device="cpu",
        context_limit=96)
    assert cap_metrics["reference_longer_than_budget"] == 2
    assert cap_metrics["duplicate_generation_records"] == 2
    assert cap_metrics["repeated_token_count"] == 2
    assert all(item["length_cap"] for item in duplicate_outputs)
    assert len(fake_cap.inputs) >= 7
    assert len(fake_cap.inputs) == 3 + 2 * 2

    class SequenceNext(torch.nn.Module):
        def __init__(self, vocab, next_ids):
            super().__init__()
            self.vocab = vocab
            self.next_ids = list(next_ids)
            self.step = 0

        def forward(self, x):
            out = torch.full((x.shape[0], x.shape[1], self.vocab), -10.0, device=x.device)
            out[:, :, self.next_ids[min(self.step, len(self.next_ids) - 1)]] = 10.0
            self.step += 1
            return out

    chinese_bytes = [token_for_byte(b) for b in "中".encode("utf-8")]
    utf8_model = SequenceNext(tok.get_vocab_size(), chinese_bytes + [eos_id])
    utf8_result = greedy_generate(utf8_model, tok, heldout[0], max_new_tokens=4)
    assert utf8_result["generated"] == "中" and utf8_result["eos"]
    partial_model = SequenceNext(tok.get_vocab_size(), [lead_e4, token_for_byte(0xB8)])
    partial_result = greedy_generate(partial_model, tok, heldout[0], max_new_tokens=2)
    assert partial_result["length_cap"] and partial_result["incomplete_utf8"]
    assert partial_result["replacement_chars"] == 0 and partial_result["generated"] == ""

    start_state = (0, 0x80, 0xBF)
    incomplete_state = utf8_state_for_bytes(b"\xe4")
    logits = torch.full((tok.get_vocab_size(),), -float("inf"))
    logits[lead_e4] = 20.0
    logits[ascii_a] = 1.0
    selected, relaxed = _select_next_id(logits, [], start_state, byte_map,
                                        special_ids, eos_id, 0.0, 1.0, 0)
    assert selected == lead_e4 and not relaxed
    logits.fill_(-float("inf"))
    logits[ascii_a] = 50.0
    logits[token_for_byte(0x80)] = 2.0
    logits[eos_id] = 100.0
    selected, relaxed = _select_next_id(logits, [], incomplete_state, byte_map,
                                        special_ids, eos_id, 0.0, 1.0, 0)
    assert selected == token_for_byte(0x80) and not relaxed

    logits.fill_(-float("inf"))
    logits[ascii_a] = 1.0
    selected, relaxed = _select_next_id(logits, [ascii_a], start_state, byte_map,
                                        special_ids, eos_id, 0.0, 1.0, 1)
    assert selected == ascii_a and relaxed, "n-gram block fallback must preserve a legal byte token"
    sample_logits = torch.arange(tok.get_vocab_size(), dtype=torch.float32)
    sample_a, _ = _select_next_id(sample_logits, [], start_state, byte_map, special_ids,
                                  eos_id, 0.7, 1.0, 0, generator=torch.Generator().manual_seed(88))
    sample_b, _ = _select_next_id(sample_logits, [], start_state, byte_map, special_ids,
                                  eos_id, 0.7, 1.0, 0, generator=torch.Generator().manual_seed(88))
    assert sample_a == sample_b, "temperature sampler must be reproducible for a fixed seed"
    order_rows = [dict(rows[0], question="问题甲"), dict(rows[1], question="问题乙"),
                  dict(rows[2], question="问题丙")]
    forward_outputs = __import__("dialogue_small").evaluate_generations(
        FixedNext(tok.get_vocab_size(), ascii_a), tok, order_rows, 6, "cpu",
        temperature=0.8, seed=41)[1]
    reverse_outputs = __import__("dialogue_small").evaluate_generations(
        FixedNext(tok.get_vocab_size(), ascii_a), tok, order_rows[::-1], 6, "cpu",
        temperature=0.8, seed=41)[1][::-1]
    assert [x["generated"] for x in forward_outputs] == [x["generated"] for x in reverse_outputs], \
        "stochastic decoding must not depend on previous question length or evaluation order"
    same_prompt_rows = [order_rows[0], dict(order_rows[0], answer="另一个参考答案")]
    same_prompt_outputs = __import__("dialogue_small").evaluate_generations(
        FixedNext(tok.get_vocab_size(), ascii_a), tok, same_prompt_rows, 6, "cpu",
        temperature=0.8, seed=41)[1]
    assert same_prompt_outputs[0]["generated"] == same_prompt_outputs[1]["generated"]
    flat_logits = torch.zeros(tok.get_vocab_size(), dtype=torch.float32)
    sampled_ids = [_select_next_id(flat_logits, [], start_state, byte_map, special_ids,
                                  eos_id, 1.0, 1.0, 0,
                                  generator=torch.Generator().manual_seed(seed))[0]
                   for seed in range(8)]
    assert len(set(sampled_ids)) > 1, "positive temperature must sample beyond deterministic argmax"
    replacement_tail = b"\xef\xbf"
    replacement_state = utf8_state_for_bytes(replacement_tail)
    replacement_id = token_for_byte(0xBD)
    replacement_logits = torch.full((tok.get_vocab_size(),), -10.0)
    replacement_logits[replacement_id] = 20.0
    continuation_bc = token_for_byte(0xBC)
    replacement_logits[continuation_bc] = 1.0
    selected, _ = _select_next_id(replacement_logits, [], replacement_state,
                                  byte_map, special_ids, eos_id, 0.0, 1.0, 0,
                                  byte_tail=replacement_tail)
    assert selected == continuation_bc, "literal U+FFFD must be suppressed by the UTF-8 replacement guard"
    repeated_logits = torch.full((tok.get_vocab_size(),), -float("inf"))
    repeated_logits[ascii_b] = 3.0
    repeated_logits[ascii_c] = 2.0
    selected, _ = _select_next_id(repeated_logits, [ascii_b], start_state, byte_map,
                                  special_ids, eos_id, 0.0, 2.0, 0)
    assert selected == ascii_c, "repetition penalty must lower a generated token's score"
    selected, _ = _select_next_id(repeated_logits, [ascii_a, ascii_b, ascii_a], start_state,
                                  byte_map, special_ids, eos_id, 0.0, 1.0, 2)
    assert selected == ascii_c, "no-repeat n-gram must block a previously observed continuation"
    assert char_f1("火星是红色", "火星是红色") == 1.0
    config_defaults = generation_config_from_args(type("Args", (), {})(), 192)
    assert config_defaults["context_limit"] == 192 and config_defaults["max_new_tokens"] == 140
    saved_config = generation_config_from_args(
        type("Args", (), {"max_new_tokens": None, "context_limit": None,
                           "temperature": None, "repetition_penalty": None,
                           "no_repeat_ngram": None, "generation_seed": None,
                           "unsafe_utf8": False})(), 192,
        dict(max_new_tokens=32, context_limit=192, temperature=0.0,
             repetition_penalty=1.0, no_repeat_ngram=0, utf8_safe=True, seed=0))
    assert saved_config["max_new_tokens"] == 32 and saved_config["context_limit"] == 192
    try:
        generation_config_from_args(type("Args", (), {"context_limit": 193})(), 192)
    except ValueError:
        pass
    else:
        raise AssertionError("generation config accepted a window larger than its training context")
    try:
        build_bpe(["fixture"], 261)
    except ValueError:
        pass
    else:
        raise AssertionError("BPE accepted a vocabulary below the ByteLevel alphabet floor")
    try:
        sample_ranges(max_bytes=16_777_217)
    except ValueError:
        pass
    else:
        raise AssertionError("range sampler accepted a request above the 16MiB ceiling")
    tr_x, tr_s, tr_l, tr_kept = encode_examples(train_rows, tok, 96, return_rows=True)
    te_x, te_s, te_l, te_kept = encode_examples(heldout, tok, 96, return_rows=True)
    assert 0 < tr_x.shape[0] == len(tr_kept) <= len(train_rows)
    assert 0 < te_x.shape[0] == len(te_kept) <= len(heldout)
    assert len(train_rows) - len(tr_kept) == sum(
        len(tok.encode(__import__("dialogue_small").format_record(r), add_special_tokens=False).ids) > 96
        for r in train_rows)
    assert len(heldout) - len(te_kept) == sum(
        len(tok.encode(__import__("dialogue_small").format_record(r), add_special_tokens=False).ids) > 96
        for r in heldout)
    assert bool((tr_s > 0).all()) and bool((te_s > 0).all())
    assert bool((tr_s < tr_l).all()) and bool((te_s < te_l).all())
    assert tok.get_vocab_size() <= 300
    assert bool((tr_x >= 0).all()) and bool((te_x >= 0).all())

    train_stream = torch.cat([tr_x[i, :int(tr_l[i])] for i in range(tr_x.shape[0])])
    pad_id = tok.token_to_id("[PAD]")
    assert not bool((train_stream == pad_id).any()), "PAD must not be counted as real corpus text"
    cfg = RCfg(vocab=tok.get_vocab_size(), d=32, n_layers=1, n_heads=4, ffn_mult=2,
               rope_dim=8, ngram_max=2, n_hash=2, n_slots=64, min_count=1,
               n_buckets=8, gate_rank=8, pos_abs_max=96, window_max=8,
               bucket_step=0)
    model = EngramRuleLM(cfg)
    build_engram_from_examples(model, tr_x, tr_l)
    total_answer_tokens = int((tr_l[:2] - tr_s[:2]).sum())
    total_first_tokens = 2
    with torch.no_grad():
        probe_logits = model(tr_x[:2])[:, :-1, :]
        probe_valid = torch.arange(tr_x.shape[1]).view(1, -1) < tr_l[:2].view(-1, 1)
        probe_token_loss, probe_first_loss, probe_tokens, probe_first_count = answer_loss_sums(
            probe_logits, tr_x[:2, 1:], probe_valid, tr_s[:2])
    assert probe_tokens == total_answer_tokens and probe_first_count == total_first_tokens
    assert torch.isfinite(probe_token_loss) and torch.isfinite(probe_first_loss)

    # Independently enumerate answer positions: target position p is predicted by logits[p-1].
    # Distinct row starts/lengths and right padding make an off-by-one mask observable.
    oracle_x = torch.tensor([[0, 1, 2, 3, 6, 6], [0, 1, 2, 3, 4, 5]])
    oracle_starts = torch.tensor([2, 3])
    oracle_lengths = torch.tensor([4, 6])
    oracle_valid = torch.arange(oracle_x.shape[1]).view(1, -1) < oracle_lengths.view(-1, 1)
    oracle_logits = (torch.arange(2 * 5 * 7, dtype=torch.float32).view(2, 5, 7) % 19) / 19
    oracle_total, oracle_first = torch.zeros(()), torch.zeros(())
    oracle_count = 0
    for row in range(oracle_x.shape[0]):
        start, length = int(oracle_starts[row]), int(oracle_lengths[row])
        row_losses = F.cross_entropy(oracle_logits[row, start - 1:length - 1],
                                     oracle_x[row, start:length], reduction="none")
        oracle_total += row_losses.sum()
        oracle_first += row_losses[0]
        oracle_count += length - start
    got_total, got_first, got_count, got_first_count = answer_loss_sums(
        oracle_logits, oracle_x[:, 1:], oracle_valid, oracle_starts)
    # Summation order differs from the per-row oracle; 1e-5 is a few float32 ulps at this scale.
    torch.testing.assert_close(got_total, oracle_total, rtol=1e-6, atol=1e-5)
    torch.testing.assert_close(got_first, oracle_first, rtol=1e-6, atol=1e-5)
    assert int(got_count) == oracle_count and int(got_first_count) == 2
    shifted = answer_loss_sums(oracle_logits, oracle_x[:, 1:], oracle_valid,
                               oracle_starts + 1)
    assert int(shifted[2]) == oracle_count - 2
    assert not torch.isclose(shifted[1], got_first, rtol=1e-6, atol=1e-5), \
        "negative control: shifting both answer starts must change first-token loss"
    for bad_valid, bad_starts in ((oracle_valid[:1], oracle_starts),
                                  (oracle_valid, oracle_starts[:1])):
        try:
            answer_loss_sums(oracle_logits, oracle_x[:, 1:], bad_valid, bad_starts)
        except ValueError:
            pass
        else:
            raise AssertionError("loss helper silently accepted a broadcastable batch shape")

    def answer_loss_for(mm, tokens, starts, lengths, first_weight):
        positions = torch.arange(tokens.shape[1]).view(1, -1)
        valid = positions < lengths.view(-1, 1)
        logits = mm(tokens)[:, :-1, :]
        token_loss, first_loss, _, _ = answer_loss_sums(logits, tokens[:, 1:], valid, starts)
        normalizer = total_answer_tokens + (first_weight - 1.0) * total_first_tokens
        return (token_loss + (first_weight - 1.0) * first_loss) / normalizer

    for first_weight in (1.0, 8.0):
        full_batch = copy.deepcopy(model).train()
        micro_batch = copy.deepcopy(model).train()
        answer_loss_for(full_batch, tr_x[:2], tr_s[:2], tr_l[:2], first_weight).backward()
        for i in range(2):
            answer_loss_for(micro_batch, tr_x[i:i + 1], tr_s[i:i + 1],
                            tr_l[i:i + 1], first_weight).backward()
        for (name, full_param), (_, micro_param) in zip(full_batch.named_parameters(),
                                                        micro_batch.named_parameters()):
            if full_param.grad is None:
                assert micro_param.grad is None, (first_weight, name)
            else:
                assert micro_param.grad is not None, (first_weight, name)
                assert float((full_param.grad - micro_param.grad).abs().max()) < 1e-5, \
                    (first_weight, name)
    # The learning-test wrapper must pass the requested onset weight through to AdamW training.
    dialogue_module = __import__("dialogue_small")
    saved_train_dialogue, saved_answer_metrics = (dialogue_module.train_dialogue,
                                                   dialogue_module.answer_metrics)
    forwarded = {}
    def capture_train_dialogue(*args, **kwargs):
        forwarded.update(kwargs)
    def capture_answer_metrics(*args, **kwargs):
        return {"examples": args[1].shape[0]}
    dialogue_module.train_dialogue = capture_train_dialogue
    dialogue_module.answer_metrics = capture_answer_metrics
    try:
        wrapped_metrics = dialogue_module.train_eval_dialogue(
            None, tr_x[:2], tr_s[:2], tr_l[:2], te_x[:2], te_s[:2], te_l[:2],
            1, 2, 1e-3, "cpu", 7, first_token_weight=8.0)
    finally:
        dialogue_module.train_dialogue = saved_train_dialogue
        dialogue_module.answer_metrics = saved_answer_metrics
    assert forwarded.get("first_token_weight") == 8.0, \
        "negative control: learning-test wrapper must not drop the requested first-token weight"
    assert wrapped_metrics[0]["examples"] == tr_x[:2].shape[0]
    assert wrapped_metrics[1]["examples"] == te_x[:2].shape[0]
    # Fixed bucket geometry must make the same query invariant to right-padding length.
    stride_cfg = RCfg(vocab=tok.get_vocab_size(), d=32, n_layers=1, n_heads=4, ffn_mult=2,
                      rope_dim=8, ngram_max=2, n_hash=2, n_slots=128, min_count=1,
                      n_buckets=8, gate_rank=8, pos_abs_max=32, window_max=4,
                      bucket_step=2, near_first=0.0)
    fixed = EngramRuleLM(stride_cfg).eval()
    fixed.engram.alive[:] = True
    dyn_cfg = RCfg(**stride_cfg.__dict__)
    dyn_cfg.bucket_step = 0
    dynamic = EngramRuleLM(dyn_cfg).eval()
    dynamic.load_state_dict(fixed.state_dict())
    for mm in (fixed, dynamic):
        for block in mm.blocks:
            with torch.no_grad():
                block.attn.bucket_proj.weight.zero_()
                block.attn.bucket_proj.bias.zero_()
                block.attn.bucket_proj.bias[1] = 2.0
    short = torch.tensor([[3, 4, 5, 6, 7, 8, 9]], dtype=torch.long)
    padded = torch.full((1, 16), tok.token_to_id("[PAD]"), dtype=torch.long)
    padded[:, :short.shape[1]] = short
    with torch.no_grad():
        fixed_short, fixed_pad = fixed(short)[0, -1], fixed(padded)[0, short.shape[1] - 1]
        dynamic_short, dynamic_pad = dynamic(short)[0, -1], dynamic(padded)[0, short.shape[1] - 1]
    fixed_diff = float((fixed_short - fixed_pad).abs().max())
    dynamic_diff = float((dynamic_short - dynamic_pad).abs().max())
    assert fixed_diff < 1e-5 and dynamic_diff > 1e-5
    prompt_rows = heldout[:3]
    forward_rows = __import__("dialogue_small").evaluate_generations(
        fixed, tok, prompt_rows, 4, "cpu")[1]
    reverse_rows = __import__("dialogue_small").evaluate_generations(
        fixed, tok, prompt_rows[::-1], 4, "cpu")[1][::-1]
    assert [r["generated"] for r in forward_rows] == [r["generated"] for r in reverse_rows]
    prompt_ids = [tok.encode(format_prefix(row), add_special_tokens=False).ids for row in prompt_rows]
    padded_prompts = torch.full((len(prompt_ids), max(map(len, prompt_ids))), pad_id)
    for i, ids in enumerate(prompt_ids):
        padded_prompts[i, :len(ids)] = torch.tensor(ids)
    with torch.no_grad():
        batch_logits = fixed(padded_prompts)
        single_logits = [fixed(torch.tensor([ids]))[0, -1] for ids in prompt_ids]
        perm = torch.tensor([2, 0, 1])
        permuted_logits = fixed(padded_prompts[perm])
    for i, ids in enumerate(prompt_ids):
        assert float((batch_logits[i, len(ids) - 1] - single_logits[i]).abs().max()) < 1e-5
    for j, i in enumerate(perm.tolist()):
        ids = prompt_ids[i]
        assert float((permuted_logits[j, len(ids) - 1] - batch_logits[i, len(ids) - 1]).abs().max()) < 1e-5
    prefix = torch.tensor([[3, 4, 5, 6, 7, 8, 9]], dtype=torch.long)
    with torch.no_grad():
        prefix_logits = fixed(prefix)[0]
        full_logits = fixed(torch.cat((prefix, torch.tensor([[10, 11, 12]], dtype=torch.long)), 1))[0, :prefix.shape[1]]
    assert float((prefix_logits - full_logits).abs().max()) < 1e-5, \
        "future tokens must not influence earlier dialogue logits"
    # A concatenated stream creates a cross-example bigram; it must not be counted by production init.
    pair = torch.tensor([[2, 3], [5, 7]], dtype=torch.long)
    pair_cfg = RCfg(vocab=16, d=16, n_layers=1, n_heads=4, ffn_mult=2, rope_dim=4,
                    ngram_max=2, n_hash=1, n_slots=4096, min_count=1, n_buckets=4,
                    gate_rank=4, pos_abs_max=16, window_max=2)
    proper, reference, joined = (EngramRuleLM(pair_cfg) for _ in range(3))
    build_engram_from_examples(proper, pair, torch.tensor([2, 2]))
    for row in pair:
        ids = reference.engram.slot_ids(row.view(1, -1)).reshape(-1)
        reference.engram.alive[ids] = True
    joined.engram.build_from_corpus(torch.tensor([[2, 3, 5, 7]]))
    assert torch.equal(proper.engram.alive, reference.engram.alive), \
        "row-wise Engram initialization differs from independent-row reference"
    assert not torch.equal(proper.engram.alive, joined.engram.alive), \
        "negative control failed to expose the cross-row n-gram"
    metrics = evaluate_dialogue(model, te_x, te_s, te_l, 2, "cpu")
    assert metrics["answer_tokens"] > 0 and metrics["examples"] == te_x.shape[0]
    metric_x, metric_s, metric_l = tr_x.repeat((17, 1)), tr_s.repeat(17), tr_l.repeat(17)
    small_batch_metrics = answer_metrics(model, metric_x, metric_s, metric_l, 7, "cpu")
    large_batch_metrics = answer_metrics(model, metric_x, metric_s, metric_l, 64, "cpu")
    assert small_batch_metrics["examples"] == large_batch_metrics["examples"] == metric_x.shape[0]
    assert small_batch_metrics["first_answer_examples"] == metric_x.shape[0]
    assert abs(small_batch_metrics["first_answer_nll"] - large_batch_metrics["first_answer_nll"]) < 1e-6
    assert small_batch_metrics["first_answer_token_accuracy"] == large_batch_metrics["first_answer_token_accuracy"]

    meta = dict(kind="modelscope_qiaojiedongfeng_dialogue_small", license=DATA_LICENSE,
                sample=dict(rows=len(rows), bytes_read=0), train_rows=len(train_rows),
                eval_rows=len(heldout), split_group="answer_text")
    train_meta = dict(steps=1, bs=2, seq=96, lr=1e-3, train_seed=7, eval_seed=0)
    with tempfile.TemporaryDirectory(prefix="dialogue-bpe-checkpoint-") as td:
        path = os.path.join(td, "small.pt")
        save_dialogue_checkpoint(path, model, cfg, tok, train_meta, meta, te_x, te_s, te_l,
                                 source_rows=rows, source_rows_complete=True)
        loaded, loaded_cfg, loaded_tok, loaded_x, loaded_s, loaded_l, payload = \
            __import__("dialogue_small").load_dialogue_checkpoint(path, "cpu")
        assert loaded_cfg.vocab == tok.get_vocab_size()
        assert loaded_tok.to_str() == tok.to_str()
        assert torch.equal(loaded_x, te_x)
        assert torch.equal(loaded_s, te_s) and torch.equal(loaded_l, te_l)
        assert payload["corpus"]["split_group"] == "answer_text"
        assert payload["tokenizer"]["kind"] == "bpe"
        assert payload["extra"]["raw_dialogue_rows_complete"] is True
        assert payload["extra"]["raw_dialogue_rows"] == rows
        again = evaluate_dialogue(loaded, loaded_x, loaded_s, loaded_l, 2, "cpu")
        assert abs(again["answer_nll"] - metrics["answer_nll"]) < 1e-6

    print("PASS question/answer component isolation, prompt-only greedy, EOS/length cap, BPE, answer mask, row-wise Engram and checkpoint round-trip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
