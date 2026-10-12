"""Small, bounded single-turn dialogue experiment for the ModelScope corpus.

No full dataset download is performed. BPE is trained only on training rows.
"""
import argparse
import hashlib
import json
import math
import os
import random
import unicodedata
from collections import Counter

import requests
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from model import EngramRuleLM, RCfg
import train as train_module
from train import DEV, evaluate, save_checkpoint, train_model

DATA_REVISION = "c0adb8dbaaeddf3b83f0e4f4cf95eb074c597610"
DATA_URL = (
    "https://www.modelscope.cn/datasets/qiaojiedongfeng/qiaojiedongfeng/"
    "resolve/" + DATA_REVISION + "/train.jsonl"
)
DATA_BYTES = 58_891_185
DATA_LICENSE = "Apache License 2.0 (dataset page declaration; file-level scope unverified)"
SPECIAL = ["[UNK]", "[PAD]", "[BOS]", "[EOS]", "[USER]", "[ASSISTANT]"]
MOJIBAKE_MARKERS = ("Ã©", "Ã¡", "Ã£", "Ã³", "Ã§", "Â ", "â€™", "â€œ",
                    "â€", "ðŸ", "ï¿½", "锟斤拷", "烫烫烫", "ä¸", "æ–",
                    "å›", "æ—", "å¤")


def record_text_fields(row):
    fields = [("question", row["question"]), ("answer", row["answer"])]
    for i, (user, assistant) in enumerate(row["history"]):
        fields.extend((("history[%d].user" % i, user),
                       ("history[%d].assistant" % i, assistant)))
    return fields


def text_quality_issues(text):
    issues = []
    if "\ufffd" in text:
        issues.append("replacement_character")
    if any(marker in text for marker in MOJIBAKE_MARKERS):
        issues.append("mojibake_signature")
    if any(ord(ch) < 0x20 and ch not in "\t\n\r" for ch in text):
        issues.append("control_character")
    return issues


def audit_text_rows(rows, tok=None):
    issues = Counter()
    roundtrip_mismatches = []
    for row_index, row in enumerate(rows):
        for field, text in record_text_fields(row):
            for issue in text_quality_issues(text):
                issues[issue] += 1
                if len(roundtrip_mismatches) < 5:
                    roundtrip_mismatches.append(dict(row=row_index, field=field, issue=issue))
        if tok is not None:
            formatted = format_record(row)
            encoded = tok.encode(formatted, add_special_tokens=False).ids
            decoded = tok.decode(encoded, skip_special_tokens=False)
            if decoded != formatted:
                issues["bpe_roundtrip_mismatch"] += 1
                if len(roundtrip_mismatches) < 5:
                    roundtrip_mismatches.append(dict(row=row_index, field="formatted_record",
                                                     issue="bpe_roundtrip_mismatch"))
    return dict(rows=len(rows), issue_counts=dict(issues),
                roundtrip_mismatches=roundtrip_mismatches,
                clean=not issues)


def require_clean_text_rows(rows, tok=None):
    audit = audit_text_rows(rows, tok)
    if not audit["clean"]:
        raise ValueError("dialogue corpus text audit failed: %s" % audit)
    return audit


def parse_record(line):
    row = json.loads(line)
    if not isinstance(row, dict):
        return None
    q = row.get("question", row.get("prompt"))
    a = row.get("answer", row.get("completion"))
    history = row.get("history", [])
    if not isinstance(q, str) or not isinstance(a, str) or not q.strip() or not a.strip():
        return None
    if not isinstance(history, list):
        return None
    turns = []
    for turn in history:
        if not isinstance(turn, (list, tuple)) or len(turn) != 2:
            return None
        if not all(isinstance(v, str) for v in turn):
            return None
        turns.append((turn[0].strip(), turn[1].strip()))
    return {"question": q.strip(), "answer": a.strip(), "history": turns}


def format_prefix(row):
    parts = ["[BOS]"]
    for user, assistant in row["history"]:
        parts.extend(("[USER]", user, "[ASSISTANT]", assistant))
    parts.extend(("[USER]", row["question"], "[ASSISTANT]"))
    return "".join(parts)


def format_record(row):
    return format_prefix(row) + row["answer"] + "[EOS]"


def sample_ranges(url=DATA_URL, total_bytes=DATA_BYTES, max_bytes=16_777_216,
                  max_lines=4096, rows_per_range=1024):
    """Read four separated bounded ranges; reject servers that ignore Range."""
    if total_bytes <= 0 or max_bytes < 4 or max_lines < 4 or rows_per_range < 1:
        raise ValueError("invalid sample limits")
    if max_bytes > 16_777_216 or max_lines > 4096 or rows_per_range > 1024:
        raise ValueError("sample request exceeds the fixed bounded-test ceiling")
    window = min(4_194_304, max_bytes // 4)
    if window < 1024:
        raise ValueError("max_bytes too small for four useful ranges")
    offsets = [((total_bytes - window) * i) // 3 for i in range(4)]
    rows, seen, ranges = [], set(), []
    rejection_counts = Counter()
    session = requests.Session()
    bytes_read = 0
    chunks_digest = hashlib.sha256()
    for start in offsets:
        end = min(total_bytes - 1, start + window - 1)
        response = session.get(url, headers={"Range": "bytes=%d-%d" % (start, end)}, timeout=30)
        if response.status_code != 206:
            raise RuntimeError("bounded dataset read requires HTTP 206 Range support; got %d" % response.status_code)
        content_range = response.headers.get("Content-Range", "")
        expected_prefix = "bytes %d-%d/" % (start, end)
        if not content_range.startswith(expected_prefix):
            raise RuntimeError("server returned an unexpected Content-Range: %s" % content_range)
        if content_range != "%s%d" % (expected_prefix, total_bytes):
            raise RuntimeError("server Content-Range total does not match the pinned dataset size")
        chunk = response.content
        if len(chunk) != end - start + 1:
            raise RuntimeError("server returned a truncated or oversized byte range")
        chunks_digest.update(start.to_bytes(8, "little"))
        chunks_digest.update(end.to_bytes(8, "little"))
        chunks_digest.update(chunk)
        bytes_read += len(chunk)
        if bytes_read > max_bytes:
            raise RuntimeError("Range sample exceeded max_bytes")
        ranges.append((start, end))
        complete = chunk.splitlines()
        if start != 0 and complete:
            complete = complete[1:]
        if end < total_bytes - 1 and complete:
            complete = complete[:-1]
        got = 0
        for raw in complete:
            if len(rows) >= max_lines or got >= rows_per_range:
                break
            try:
                text = raw.decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                rejection_counts["invalid_utf8"] += 1
                continue
            try:
                row = parse_record(text)
            except json.JSONDecodeError:
                rejection_counts["invalid_json"] += 1
                continue
            if row is None:
                rejection_counts["invalid_schema"] += 1
                continue
            text_issues = [issue for _, field_text in record_text_fields(row)
                           for issue in text_quality_issues(field_text)]
            if text_issues:
                for issue in text_issues:
                    rejection_counts[issue] += 1
                continue
            key = (row["question"], row["answer"], tuple(row["history"]))
            if key in seen:
                rejection_counts["duplicate_row"] += 1
                continue
            seen.add(key)
            rows.append(row)
            got += 1
    if len(rows) < 128:
        raise RuntimeError("bounded sample produced only %d valid rows; required at least 128" % len(rows))
    multi_turn_rows = sum(bool(r["history"]) for r in rows)
    text_audit = require_clean_text_rows(rows)
    row_digest = hashlib.sha256()
    for row in rows:
        row_digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8"))
        row_digest.update(b"\n")
    return rows, dict(bytes_read=bytes_read, ranges=ranges, rows=len(rows),
                      multi_turn_rows=multi_turn_rows, history_available=multi_turn_rows > 0,
                      max_bytes=max_bytes, max_lines=max_lines, source_url=url,
                      revision=DATA_REVISION, license=DATA_LICENSE,
                      text_audit=text_audit, sample_sha256=row_digest.hexdigest(),
                      range_sha256=chunks_digest.hexdigest(),
                      rejection_counts=dict(rejection_counts),
                      schema="question/answer; history optional; single-turn sample validated")


def bytelevel_token_bytes(tok):
    """Map ByteLevel visible-token characters back to their original bytes."""
    byte_values = list(range(ord("!"), ord("~") + 1))
    byte_values += list(range(ord("¡"), ord("¬") + 1))
    byte_values += list(range(ord("®"), ord("ÿ") + 1))
    codepoints = list(byte_values)
    extra = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            codepoints.append(256 + extra)
            extra += 1
    decoder = {chr(codepoint): value for value, codepoint in zip(byte_values, codepoints)}
    special_ids = set(tok.get_added_tokens_decoder().keys())
    special_ids.update(tok.token_to_id(s) for s in SPECIAL if tok.token_to_id(s) is not None)
    mapping = {}
    invalid = []
    for token_id in range(tok.get_vocab_size()):
        token = tok.id_to_token(token_id)
        if token_id in special_ids:
            mapping[token_id] = None
            continue
        try:
            mapping[token_id] = bytes(decoder[ch] for ch in token)
        except KeyError:
            invalid.append((token_id, token))
    if invalid:
        raise ValueError("ByteLevel vocab contains unmapped token chars: %r" % (invalid[:4],))
    return mapping, special_ids


def build_bpe(train_texts, vocab_size=2048):
    if vocab_size <= len(SPECIAL) or not train_texts:
        raise ValueError("BPE needs training texts and vocab_size greater than special-token count")
    corrupt = [(i, text_quality_issues(text)) for i, text in enumerate(train_texts)
               if text_quality_issues(text)]
    if corrupt:
        raise ValueError("refusing to train ByteLevel BPE on suspicious text: %r" % (corrupt[:5],))
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    min_vocab = 256 + len(SPECIAL)
    if vocab_size < min_vocab:
        raise ValueError("ByteLevel BPE vocab_size must be at least %d" % min_vocab)
    tok.train_from_iterator(train_texts, trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=2, special_tokens=SPECIAL,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    if tok.get_vocab_size() > vocab_size:
        raise AssertionError("BPE trainer exceeded the requested vocabulary size")
    return tok


def build_split_bpe(train_rows, eval_rows, vocab_size=512):
    if not train_rows or not eval_rows:
        raise ValueError("BPE fitting requires nonempty train and eval splits")
    require_clean_text_rows(train_rows)
    require_clean_text_rows(eval_rows)
    tokenizer = build_bpe([format_record(row) for row in train_rows], vocab_size)
    train_audit = audit_text_rows(train_rows, tokenizer)
    eval_audit = audit_text_rows(eval_rows, tokenizer)
    if not train_audit["clean"] or not eval_audit["clean"]:
        raise ValueError("split tokenizer failed BPE round-trip audit")
    return tokenizer, dict(train=train_audit, eval=eval_audit,
                           fit_rows=len(train_rows), eval_rows=len(eval_rows),
                           fit_scope="train_only")


def encode_examples(rows, tok, max_len, return_rows=False):
    ids, answer_starts, lengths, kept_rows = [], [], [], []
    for row in rows:
        enc = tok.encode(format_record(row), add_special_tokens=False)
        prefix_ids = tok.encode(format_prefix(row), add_special_tokens=False).ids
        f = enc.ids
        if f[:len(prefix_ids)] != prefix_ids:
            raise ValueError("BPE prompt encoding is not a prefix of full question/answer encoding")
        answer_start = len(prefix_ids)
        if len(f) > max_len:
            continue
        if len(f) < 3 or answer_start < 3 or answer_start >= len(f):
            continue
        ids.append(f)
        answer_starts.append(answer_start)
        lengths.append(len(f))
        kept_rows.append(row)
    if not ids:
        raise ValueError("all encoded dialogue rows were too short or invalid")
    pad = tok.token_to_id("[PAD]")
    x = torch.full((len(ids), max_len), pad, dtype=torch.long)
    starts = torch.zeros(len(ids), dtype=torch.long)
    lens = torch.tensor(lengths, dtype=torch.long)
    for i, seq in enumerate(ids):
        x[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        starts[i] = answer_starts[i]
    result = (x, starts, lens)
    return result + (kept_rows,) if return_rows else result


def partition_by_token_budget(rows, tok, max_tokens):
    kept, rejected = [], []
    for row in rows:
        length = len(tok.encode(format_record(row), add_special_tokens=False).ids)
        if length <= max_tokens:
            kept.append(row)
        else:
            rejected.append(dict(question=row["question"], token_length=length))
    return kept, dict(limit=max_tokens, rejected=len(rejected),
                      max_rejected_tokens=max((r["token_length"] for r in rejected), default=0),
                      examples=rejected[:8])


def required_sequence_length(rows, tok, percentile=1.0):
    if not rows or not 0.5 <= percentile <= 1.0:
        raise ValueError("sequence sizing requires rows and percentile in [0.5,1]")
    lengths = sorted(len(tok.encode(format_record(row), add_special_tokens=False).ids)
                     for row in rows)
    index = min(len(lengths) - 1, math.ceil(percentile * len(lengths)) - 1)
    return lengths[index], dict(max_length=max(lengths), percentile=percentile,
                                percentile_length=lengths[index], rows=len(lengths))


def batches(x, starts, lengths, batch_size, device, shuffle, seed, trim_padding=False):
    """trim_padding=True: 每个批只算到该批**真实最大行长**(等价性论证见 `_trim_length`)。

    ★只对 `bucket_step > 0`(固定步长)安全 —— legacy 动态步长下 `_bucket_step(L)` 随 L 变,
      截断会改变每个位置的窗口。调用方负责判这个前提。
    """
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(x.shape[0], generator=g) if shuffle else torch.arange(x.shape[0])
    for ids in order.split(batch_size):
        bx = x[ids].to(device)
        bs = starts[ids].to(device)
        lens = lengths[ids].to(device)
        if trim_padding:
            bx = _trim_length(bx, lens)
        pos = torch.arange(bx.shape[1], device=device).view(1, -1)
        valid = pos < lens.view(-1, 1)
        target = bx[:, 1:]
        target_pos = pos[:, 1:]
        mask = valid[:, 1:] & (target_pos >= bs.view(-1, 1))
        yield bx, target, mask


@torch.no_grad()
def build_engram_from_examples(model, x, lengths):
    """Count only real tokens within each row; never create PAD or cross-row n-grams."""
    counts = torch.zeros(model.c.n_slots, dtype=torch.long, device=model.engram.alive.device)
    for i, length in enumerate(lengths.tolist()):
        if length < 2:
            continue
        ids = model.engram.slot_ids(x[i:i + 1, :length].to(model.engram.alive.device)).reshape(-1)
        counts.index_add_(0, ids, torch.ones_like(ids, dtype=torch.long))
    model.engram.alive.copy_(counts >= model.c.min_count)
    return dict(alive_slots=int(model.engram.alive.sum()), n_slots=model.c.n_slots,
                alive_ratio=float(model.engram.alive.float().mean()))


def answer_loss_sums(logits, target, valid, starts):
    if (logits.ndim != 3 or target.ndim != 2 or valid.ndim != 2 or starts.ndim != 1 or
            logits.shape[:2] != target.shape or
            valid.shape != (target.shape[0], target.shape[1] + 1) or
            starts.shape[0] != target.shape[0]):
        raise ValueError("answer loss expects logits[B,T,V], target[B,T], valid[B,T+1], starts[B]")
    positions = torch.arange(1, target.shape[1] + 1, device=target.device).view(1, -1)
    answer_mask = valid[:, 1:] & (positions >= starts.view(-1, 1))
    first_mask = answer_mask & (positions == starts.view(-1, 1))
    losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                             target.reshape(-1), reduction="none").view_as(target)
    return ((losses * answer_mask).sum(), (losses * first_mask).sum(),
            answer_mask.sum(), first_mask.sum())


def _trim_length(bx, lengths_b, min_len=2):
    """把一个微批的张量截到该批**真实最大行长**, 省掉尾部补齐位的全部算力。

    ★为什么这是**精确等价**而不是近似(§51.9 实测: 820 行训练数据里 86.2% 的位置
      是补齐位, 中位行长 60 / 张量长度 512):
      1. 补齐位在序列**尾部**(positions ≥ length)。因果注意力 ⇒ 尾部位**不可能**影响
         它前面的任何位置 ⇒ 截掉不改变任何保留位置的隐状态。
      2. loss 侧本来就有 `valid = pos < lengths` 把它们 mask 掉了。
      3. Engram 分支只读每个位置自己的 n-gram, 不跨位置。
      ⇒ 保留前缀的前向与 loss 完全相同; 只有浮点归约形状变化带来的机器精度差异。

    ★★**前提**: 只在 `bucket_step > 0`(固定步长)时成立。`bucket_step == 0` 是 legacy
      动态步长, `_bucket_step(L) = max(1, ceil(L/n_buckets))` **随 L 变** ⇒ 截断会改变
      **每一个位置**的窗口, 不等价。调用方必须自己判这个前提(见 train_dialogue)。
    """
    keep = int(lengths_b.max())
    return bx[:, :max(keep, min_len)]


def train_dialogue(model, x, starts, lengths, steps, batch_size, lr, device, seed=7,
                   first_token_weight=1.0, micro_batch=None, trim_padding=True):
    """micro_batch=None 时行为与历史完全一致(规则 3: 不得顺手改出货训练路径)。

    显式给 micro_batch 只用于**显存兜底**: 窗口支路会物化 (B,L,W,H,head_dim) 中间张量
    (§40), 其大小 ∝ micro_batch × d, d 放大 4 倍就会打满显存。
    ★切分微批**不改变**累积梯度 —— 每个微批都除以同一个 loss_normalizer, 梯度是相加的,
      `sum(部分) == sum(整体)`, 只有浮点求和顺序不同(规则 14 的机器精度)。
      因此这是**精确**等价的, 不是近似。

    ★trim_padding=True: 把每个微批截到该批真实最大行长(见 `_trim_length` 的等价性论证)。
      训练实测 profile 显示这一步 95% 的时间是 GPU kernel 忙时(不是提交开销), 而
      86.2% 的位置是补齐位 ⇒ **这是唯一能动到真实算力的一条**。
      ★只在 `bucket_step > 0` 时启用; legacy 动态步长下自动退回不截断(否则会改窗口)。
    """
    if (x.shape[0] < 2 or x.shape[1] < 4 or steps < 1 or batch_size < 1 or lr <= 0 or
            not math.isfinite(first_token_weight) or first_token_weight < 1.0):
        raise ValueError("invalid small-dialogue training settings")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    gen = torch.Generator().manual_seed(seed)
    model.train()
    micro_batch_size = (max(1, min(64, 32_768 // x.shape[1])) if micro_batch is None
                        else int(micro_batch))
    if not 1 <= micro_batch_size <= batch_size:
        raise ValueError("micro_batch must be within [1, batch_size], got %d" % micro_batch_size)
    # ★截断只对固定步长安全(见 _trim_length)。legacy 动态步长下自动关闭并说明原因。
    trim = bool(trim_padding) and getattr(model.c, "bucket_step", 0) > 0
    for step in range(steps):
        indices = torch.randint(0, x.shape[0], (batch_size,), generator=gen)
        answer_tokens = (lengths[indices] - starts[indices]).sum().clamp_min(1).to(device)
        loss_normalizer = answer_tokens + (first_token_weight - 1.0) * len(indices)
        opt.zero_grad(set_to_none=True)
        step_loss = torch.zeros((), dtype=torch.float32, device=device)
        for i in indices.split(micro_batch_size):
            bx = x[i].to(device)
            starts_b = starts[i].to(device)
            lengths_b = lengths[i].to(device)
            if trim:
                bx = _trim_length(bx, lengths_b)
            pos = torch.arange(bx.shape[1], device=device).view(1, -1)
            valid = pos < lengths_b.view(-1, 1)
            logits = model(bx)[:, :-1, :]
            target = bx[:, 1:]
            token_loss, first_loss, _, _ = answer_loss_sums(logits, target, valid, starts_b)
            loss_sum = token_loss + (first_token_weight - 1.0) * first_loss
            (loss_sum / loss_normalizer).backward()
            step_loss += loss_sum.detach()
        opt.step()
        sched.step()
        if step == 0 or (step + 1) % max(1, steps // 4) == 0:
            print("train step %d/%d weighted_answer_loss %.4f" %
                  (step + 1, steps, float(step_loss / loss_normalizer)))
    model.eval()


@torch.no_grad()
def evaluate_dialogue(model, x, starts, lengths, batch_size, device, trim_padding=True):
    model.eval()
    # ★与 train_dialogue 同一条守卫: 只有固定步长下截断才等价(见 _trim_length)
    trim = bool(trim_padding) and getattr(model.c, "bucket_step", 0) > 0
    total_loss = torch.zeros((), device=device)
    total_tokens = torch.zeros((), dtype=torch.long, device=device)
    correct = torch.zeros((), dtype=torch.long, device=device)
    first_loss = torch.zeros((), device=device)
    first_correct = torch.zeros((), dtype=torch.long, device=device)
    first_count = torch.zeros((), dtype=torch.long, device=device)
    examples = 0
    for bx, target, mask in batches(x, starts, lengths, batch_size, device,
                                    shuffle=False, seed=0, trim_padding=trim):
        logits = model(bx)[:, :-1, :]
        batch_starts = starts[examples:examples + bx.shape[0]].to(device)
        batch_lengths = lengths[examples:examples + bx.shape[0]].to(device)
        valid = torch.arange(bx.shape[1], device=device).view(1, -1) < batch_lengths.view(-1, 1)
        token_loss, onset_loss, n_tokens, n_first = answer_loss_sums(
            logits, target, valid, batch_starts)
        total_loss += token_loss
        first_loss += onset_loss
        total_tokens += n_tokens
        first_count += n_first
        correct += ((logits.argmax(-1) == target) & mask).sum()
        batch_index = torch.arange(bx.shape[0], device=device)
        first_pos = batch_starts - 1
        first_correct += (logits[batch_index, first_pos].argmax(-1) ==
                          bx[batch_index, batch_starts]).sum()
        examples += bx.shape[0]
    if not bool(total_tokens) or not bool(first_count):
        raise ValueError("no answer tokens in evaluation split")
    answer_nll = float(total_loss / total_tokens)
    first_answer_nll = float(first_loss / first_count)
    return dict(answer_nll=answer_nll,
                answer_ppl=math.exp(min(answer_nll, 20)),
                answer_token_accuracy=float(correct / total_tokens),
                answer_tokens=int(total_tokens),
                first_answer_nll=first_answer_nll,
                first_answer_token_accuracy=float(first_correct / first_count),
                first_answer_examples=int(first_count), examples=examples)


def utf8_step(state, byte):
    """Advance a UTF-8 DFA state; None means the prefix is irrecoverably invalid."""
    remaining, low, high = state
    if remaining:
        if not low <= byte <= high:
            return None
        remaining -= 1
        return (remaining, 0x80, 0xBF) if remaining else (0, 0x80, 0xBF)
    if byte <= 0x7F:
        return (0, 0x80, 0xBF)
    if 0xC2 <= byte <= 0xDF:
        return (1, 0x80, 0xBF)
    if byte == 0xE0:
        return (2, 0xA0, 0xBF)
    if 0xE1 <= byte <= 0xEC or 0xEE <= byte <= 0xEF:
        return (2, 0x80, 0xBF)
    if byte == 0xED:
        return (2, 0x80, 0x9F)
    if byte == 0xF0:
        return (3, 0x90, 0xBF)
    if 0xF1 <= byte <= 0xF3:
        return (3, 0x80, 0xBF)
    if byte == 0xF4:
        return (3, 0x80, 0x8F)
    return None


def utf8_state_for_bytes(data):
    state = (0, 0x80, 0xBF)
    for byte in data:
        state = utf8_step(state, byte)
        if state is None:
            return None
    return state


def valid_token_bytes(state, fragment):
    next_state = state
    for byte in fragment:
        next_state = utf8_step(next_state, byte)
        if next_state is None:
            return None
    return next_state


class _DecodeMask:
    """utf8-safe **允许集**的状态查表: 构造一次, 之后每步只剩"查表 + 一次 masked_fill"。

    ★ §51 性能: 原实现在 `_select_next_id` 里**每步**用 Python 重算一遍 allowed 列表,
      再对 vocab 个位置逐个做张量索引赋值(每步 vocab 次 kernel 提交)。实测
      vocab=1024 时选择侧占生成 host 时间的 35%, 且单次调用有 94 次 device→host 同步。

    等价性来源: 表里每格都是"用原判定(valid_token_bytes + U+FFFD 守卫)对该状态下
    每个 token 算一遍"的结果, 只是把每步重算换成查表 —— 见 `check_fast_path_equiv.py`
    的 E3(279 用例 × CPU/CUDA 全电池)与 E4(端到端 token 序列逐条一致)。

    状态集合不是硬编码的: 从起始状态 BFS, 只覆盖**用该 token 器的字节片段真正可达**的
    UTF-8 状态; 查不到的 state 直接报错(而不是悄悄当成起始状态)。
    """

    MAX_TAILS = 4096          # ★ 规则 9: 记忆化必须有硬上限

    def __init__(self, byte_map, eos_id, device):
        self.byte_map = byte_map
        self.eos_id = eos_id
        self.device = device
        self.V = (max(byte_map) + 1) if byte_map else 0
        if self.V <= 0:
            raise ValueError("byte_map is empty: cannot build the decoder mask table")
        start = (0, 0x80, 0xBF)
        states = [start]
        index = {start: 0}
        rows = []
        head = 0
        while head < len(states):
            st = states[head]
            row = torch.zeros(self.V, dtype=torch.bool)
            for tid, frag in byte_map.items():
                if frag is None:                      # 特殊 token: 只在起始状态放行 EOS
                    if tid == eos_id and st == start:
                        row[tid] = True
                    continue
                nxt = valid_token_bytes(st, frag)
                if nxt is None:
                    continue
                row[tid] = True
                if nxt not in index:
                    index[nxt] = len(states)
                    states.append(nxt)
            rows.append(row)
            head += 1
        assert head == len(states), "decode-mask BFS left states unprocessed"
        empty = [i for i, r in enumerate(rows) if not bool(r.any())]
        assert not empty, ("decoder constraints left no legal next token in states %s; "
                           "the tokenizer cannot finish these partial UTF-8 sequences"
                           % [states[i] for i in empty])
        self.state_row = {st: i for i, st in enumerate(states)}
        self.table = torch.stack(rows).to(device)       # (S, V) bool
        self._tails = {}

    def _tail_row(self, byte_tail):
        row = self._tails.get(byte_tail)
        if row is None:
            if len(self._tails) >= self.MAX_TAILS:
                self._tails.clear()                    # 超上限就清空(不影响正确性)
            ok = torch.ones(self.V, dtype=torch.bool)
            if byte_tail:
                for tid, frag in self.byte_map.items():
                    if frag is not None and b"\xef\xbf\xbd" in (byte_tail + frag):
                        ok[tid] = False
            row = ok.to(self.device)
            self._tails[byte_tail] = row
        return row

    def allowed_mask(self, byte_state, byte_tail=b""):
        i = self.state_row.get(byte_state)
        if i is None:
            raise RuntimeError("UTF-8 state %r is unreachable from this tokenizer's byte "
                               "fragments; the mask table would silently mis-allow tokens"
                               % (byte_state,))
        mask = self.table[i]
        if byte_tail:
            mask = mask & self._tail_row(byte_tail)
        return mask


def _apply_repetition_penalty(scores, history, eos_id, repetition_penalty):
    """重复惩罚(向量化)。与逐元素版本等价: 非有限项乘/除仍是 ±inf, 语义不变。"""
    if repetition_penalty <= 1.0:
        return scores
    hist = sorted({t for t in history if t != eos_id})
    if not hist:
        return scores
    idx = torch.tensor(hist, dtype=torch.long, device=scores.device)
    s = scores[idx]
    scores[idx] = torch.where(s > 0, s / repetition_penalty, s * repetition_penalty)
    return scores


def _select_next_id(logits, history, byte_state, byte_map, special_ids,
                    eos_id, temperature, repetition_penalty, no_repeat_ngram,
                    utf8_safe=True, generator=None, byte_tail=b"", mask_cache=None):
    """★ §51: 允许集查表 + 惩罚/禁重复向量化, 每步同步由 94 次降到 1 次(贪心路径)。
    语义逐位不变 —— 判据见 `check_fast_path_equiv.py` (E3 全电池 / E4 端到端)。"""
    scores = logits.float().clone()
    scores[~torch.isfinite(scores)] = -float("inf")
    allowed = None
    if utf8_safe:
        mc = mask_cache if mask_cache is not None else _DecodeMask(byte_map, eos_id,
                                                                   scores.device)
        allowed = mc.allowed_mask(byte_state, byte_tail)
        scores = scores.masked_fill(~allowed, -float("inf"))
    _apply_repetition_penalty(scores, history, eos_id, repetition_penalty)
    if no_repeat_ngram > 0 and len(history) >= no_repeat_ngram - 1:
        prefix = tuple(history[-(no_repeat_ngram - 1):]) if no_repeat_ngram > 1 else ()
        # 掩一个本来就被 allowed 挡掉的 token 是空操作 ⇒ 不必再查 allowed_set
        blocked = sorted({history[i + no_repeat_ngram - 1]
                          for i in range(len(history) - no_repeat_ngram + 1)
                          if tuple(history[i:i + no_repeat_ngram - 1]) == prefix
                          and history[i + no_repeat_ngram - 1] != eos_id})
        if blocked:
            scores[torch.tensor(blocked, dtype=torch.long, device=scores.device)] = \
                -float("inf")
    mx, top = scores.max(0)
    if temperature == 0:
        # ★ 一次同步同时取回"argmax"和"是否还有有限候选"(cat+tolist 只触发一次 D2H)
        sel, finite = torch.cat([top.reshape(1),
                                 torch.isfinite(mx).reshape(1).long()]).tolist()
        if finite:
            return sel, False
        # 约束把合法候选全压成 -inf ⇒ 松弛: 退回"只按 allowed 掩码"重算一次
        scores = logits.float().clone()
        scores[~torch.isfinite(scores)] = -float("inf")
        if allowed is not None:
            scores = scores.masked_fill(~allowed, -float("inf"))
        _apply_repetition_penalty(scores, history, eos_id, repetition_penalty)
        if not bool(torch.isfinite(scores.max())):
            raise RuntimeError("all legal decoder candidates have non-finite logits")
        return int(scores.argmax()), True
    probs = torch.softmax(scores / temperature, dim=-1)
    if not bool(torch.isfinite(probs).all()) or float(probs.sum()) <= 0:
        raise RuntimeError("generation sampler produced invalid probabilities")
    return int(torch.multinomial(probs, num_samples=1, generator=generator)), False


def _finalize_generation(row, prompt_ids, generated_ids, reference_tokens,
                         generation_budget, max_new_tokens, context_budget,
                         byte_state, answer_ids, hit_eos, constraint_relaxed_count,
                         eos_id, byte_map, tok, utf8_safe):
    """生成结果的记账。**逐行与批生成共用同一份** —— 这样"两条路径字段一致"不是靠
    对齐两份代码, 而是结构上只有一份(E5 检查量的是这个函数的输出)。"""
    content_ids = [i for i in generated_ids if i != eos_id]
    incomplete_utf8 = bool(utf8_safe and byte_state != (0, 0x80, 0xBF))
    if utf8_safe:
        raw_bytes = b"".join(byte_map[i] for i in content_ids)
        try:
            generated = raw_bytes.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            if exc.reason != "unexpected end of data" or not incomplete_utf8:
                raise AssertionError("UTF-8-safe generator produced invalid bytes") from exc
            generated = raw_bytes[:exc.start].decode("utf-8", errors="strict")
    else:
        generated = tok.decode(content_ids, skip_special_tokens=True)
    repeated_tokens = len(answer_ids) - len(set(answer_ids))
    trigram_counts = Counter(tuple(answer_ids[i:i + 3])
                             for i in range(max(0, len(answer_ids) - 2)))
    repeated_trigrams = sum(count - 1 for count in trigram_counts.values() if count > 1)
    generated = generated.strip()
    length_cap = not hit_eos and len(generated_ids) >= generation_budget
    return dict(question=row["question"], reference=row["answer"], generated=generated,
                prompt_tokens=len(prompt_ids), generated_tokens=len(generated_ids),
                generation_budget=generation_budget, reference_tokens=reference_tokens,
                # ★ §54: **只增不改**的出口 —— 逐 token 的答案 id(不含 EOS/filler)。
                #   之前只有解码后的文本, 于是"自由生成与教师强制在第几个 token 分歧"
                #   这类测量只能拿文本重新 tokenize(BPE 不保证还原原 id 序列)。
                #   逐行与批生成共用本函数 ⇒ 两条路径的 id 天然同源。
                answer_ids=list(answer_ids),
                eos=hit_eos, truncated_context=length_cap and context_budget < max_new_tokens,
                length_cap=length_cap,
                incomplete_utf8=incomplete_utf8, replacement_chars=generated.count("�"),
                constraint_relaxed_steps=constraint_relaxed_count,
                repeated_token_count=repeated_tokens,
                repeated_token_rate=repeated_tokens / max(1, len(answer_ids)),
                repeated_trigram_count=repeated_trigrams,
                repeated_trigram_rate=repeated_trigrams / max(1, len(answer_ids) - 2))


@torch.no_grad()
def greedy_generate(model, tok, row, max_new_tokens=140, device="cpu", context_limit=96,
                    temperature=0.0, repetition_penalty=1.0, no_repeat_ngram=0,
                    utf8_safe=True, generator=None, mask_cache=None):
    prompt = format_prefix(row)
    prompt_ids = tok.encode(prompt, add_special_tokens=False).ids
    reference_ids = tok.encode(format_record(row), add_special_tokens=False).ids
    if (reference_ids[:len(prompt_ids)] != prompt_ids or
            len(reference_ids) <= len(prompt_ids)):
        raise ValueError("generation reference does not extend the encoded prompt")
    reference_tokens = len(reference_ids) - len(prompt_ids)
    if not prompt_ids or max_new_tokens < 1 or context_limit < 1:
        raise ValueError("generation needs a nonempty prompt and positive length limits")
    if len(prompt_ids) > context_limit:
        raise ValueError("prompt exceeds context_limit; refusing to reset its absolute positions")
    if (not math.isfinite(temperature) or temperature < 0 or
            not math.isfinite(repetition_penalty) or repetition_penalty < 1.0 or
            no_repeat_ngram < 0):
        raise ValueError("invalid generation temperature/repetition settings")
    ids = list(prompt_ids)
    context_budget = context_limit - len(prompt_ids)
    generation_budget = min(max_new_tokens, context_budget)
    eos_id = tok.token_to_id("[EOS]")
    if eos_id is None:
        raise ValueError("tokenizer is missing the required [EOS] token")
    byte_map, special_ids = bytelevel_token_bytes(tok)
    # ★ §51: 允许集查表构造一次(多行评测请把同一个对象传进来复用, 见 evaluate_generations)
    if utf8_safe and mask_cache is None:
        mask_cache = _DecodeMask(byte_map, eos_id, device)
    byte_state = (0, 0x80, 0xBF)
    answer_ids = []
    hit_eos = False
    constraint_relaxed_count = 0
    byte_tail = b""
    model.eval()
    for _ in range(generation_budget):
        # Keep the original position origin used by teacher-forced training. Sliding a
        # suffix to position zero changes RoPE/absolute-position features after the cap.
        x = torch.tensor(ids, dtype=torch.long, device=device).view(1, -1)
        # ★§61/§62: 每步都把 model 上挂着的 q_pool 一起喂进去。
        #   不喂的话 forward(x) 里的 q_pool=None 会走"模型自切", 而自切用的是**当批**
        #   的 emb ⇒ 与外部设的 q_pool 无关 ⇒ 外部设的值在这条路径上完全没用上
        #   (实测: set_question_pool(ZERO) 与 set_question_pool(RAND) 逐 token 相同)。
        #   ★getattr 兜底: 测试用的假模型(FixedNext 等)没有这个方法 ⇒ 必须退回旧行为,
        #     否则"给出货路径加可选参数"会顺带打破全部单测。
        qp = getattr(model, "q_pool", None)
        logits = model(x)[0, -1] if qp is None else model(x, q_pool=qp)[0, -1]
        next_id, relaxed = _select_next_id(
            logits, answer_ids, byte_state, byte_map, special_ids, eos_id,
            temperature, repetition_penalty, no_repeat_ngram, utf8_safe, generator,
            byte_tail, mask_cache)
        constraint_relaxed_count += int(relaxed)
        ids.append(next_id)
        if utf8_safe and next_id != eos_id:
            fragment = byte_map[next_id]
            byte_state = valid_token_bytes(byte_state, fragment)
            if byte_state is None:
                raise AssertionError("selected ByteLevel token violated UTF-8 state")
            byte_tail = (byte_tail + fragment)[-2:]
        if next_id == eos_id:
            hit_eos = True
            break
        answer_ids.append(next_id)
    generated_ids = ids[len(prompt_ids):]
    return _finalize_generation(row, prompt_ids, generated_ids, reference_tokens,
                                generation_budget, max_new_tokens, context_budget,
                                byte_state, answer_ids, hit_eos, constraint_relaxed_count,
                                eos_id, byte_map, tok, utf8_safe)


@torch.no_grad()
def greedy_generate_batch(model, tok, rows, max_new_tokens=140, device="cpu",
                          context_limit=96, temperature=0.0, repetition_penalty=1.0,
                          no_repeat_ngram=0, utf8_safe=True, mask_cache=None,
                          generators=None, q_pool=None):
    """★ §51: **等长分组**批生成。**模型一行都不用改**。

    为什么能不改: 只把 **prompt 长度相同**的行放进同一批 ⇒ 批内每行 prompt 等长 ⇒
    生成预算相同 ⇒ 每行序列长度始终相等 ⇒ pos=arange(L) 对**每一行**都成立,
    不需要左填充, 也不必把 pos 打通成 per-row。(左填充会把真实 token 挤到位置 p..L-1,
    与 teacher-forced 训练时的位置原点不一致 —— 那正是 greedy_generate 注释里
    拒绝滑窗重置位置的理由。)
    已出 EOS 的行用 EOS 补齐长度以保持张量形状, 其后续 token 全部丢弃。

    前向每步只提交一次(G 行 vs 逐行的 G 次): profile 实测 L=96 前向 B=1 5.42 ms、
    B=16 5.99 ms ⇒ 每行前向成本降到 6.9%。逐行等价判据见 check_fast_path_equiv.py 的 E5。
    """
    if not rows:
        raise ValueError("batch generation requires at least one row")
    if (not math.isfinite(temperature) or temperature < 0 or
            not math.isfinite(repetition_penalty) or repetition_penalty < 1.0 or
            no_repeat_ngram < 0):
        raise ValueError("invalid generation temperature/repetition settings")
    if max_new_tokens < 1 or context_limit < 1:
        raise ValueError("generation needs positive length limits")
    eos_id = tok.token_to_id("[EOS]")
    if eos_id is None:
        raise ValueError("tokenizer is missing the required [EOS] token")
    byte_map, special_ids = bytelevel_token_bytes(tok)
    if utf8_safe and mask_cache is None:
        mask_cache = _DecodeMask(byte_map, eos_id, device)

    meta = []
    for row in rows:
        prompt_ids = tok.encode(format_prefix(row), add_special_tokens=False).ids
        reference_ids = tok.encode(format_record(row), add_special_tokens=False).ids
        if (reference_ids[:len(prompt_ids)] != prompt_ids or
                len(reference_ids) <= len(prompt_ids)):
            raise ValueError("generation reference does not extend the encoded prompt")
        if not prompt_ids:
            raise ValueError("generation needs a nonempty prompt")
        if len(prompt_ids) > context_limit:
            raise ValueError("prompt exceeds context_limit; refusing to reset its absolute positions")
        context_budget = context_limit - len(prompt_ids)
        meta.append(dict(row=row, prompt_ids=prompt_ids,
                         reference_tokens=len(reference_ids) - len(prompt_ids),
                         context_budget=context_budget,
                         generation_budget=min(max_new_tokens, context_budget),
                         byte_state=(0, 0x80, 0xBF), answer_ids=[], hit_eos=False,
                         relaxed=0, byte_tail=b"", done=False))

    by_len = {}
    for i, m in enumerate(meta):
        by_len.setdefault(len(m["prompt_ids"]), []).append(i)
    out = [None] * len(meta)
    model.eval()
    qpool_groups = None
    if q_pool is not None:
        # ★按 prompt 长度分组后, q_pool 也必须**跟着重排** —— 组内第 g 行对应
        #   idxs[g], 所以它的 q_pool 在原数组里的下标就是 idxs[g]。
        assert q_pool.shape[0] == len(rows), \
            "q_pool 的行数(%d)必须等于 rows(%d)" % (q_pool.shape[0], len(rows))
        qpool_groups = {_plen: q_pool[idxs] for _plen, idxs in by_len.items()}
    for _plen, idxs in by_len.items():
        G = len(idxs)
        qp = None if qpool_groups is None else qpool_groups[_plen]
        ids_mat = [list(meta[i]["prompt_ids"]) for i in idxs]
        budget = min(meta[i]["generation_budget"] for i in idxs)
        gens = [generators[i] if generators is not None else None for i in idxs]
        n_live = G
        for _ in range(budget):
            if n_live == 0:
                break
            x = torch.tensor(ids_mat, dtype=torch.long, device=device)
            # ★ §61/§62: 逐组传 q_pool。`qpool_groups[_plen]` 已经是**按本组行序**排好的
            #   (它是 q_pool[idxs], idxs 正是本组的行下标) ⇒ 直接整块传, 不需要再切片。
            #   (第一版写成 qp[g0:g0+G] 引用了不存在的 g0 ⇒ NameError。)
            #   None ⇒ 模型自己从 x 里切问题段(前提 cfg 给了 token id)。
            logits = model(x)[:, -1, :] if qp is None \
                else model(x, q_pool=qp)[:, -1, :]
            for g, i in enumerate(idxs):
                m = meta[i]
                if m["done"]:
                    ids_mat[g].append(eos_id)        # 只为对齐长度; 记账时剔掉
                    continue
                next_id, relaxed = _select_next_id(
                    logits[g], m["answer_ids"], m["byte_state"], byte_map, special_ids,
                    eos_id, temperature, repetition_penalty, no_repeat_ngram, utf8_safe,
                    gens[g], m["byte_tail"], mask_cache)
                m["relaxed"] += int(relaxed)
                ids_mat[g].append(next_id)
                if utf8_safe and next_id != eos_id:
                    fragment = byte_map[next_id]
                    m["byte_state"] = valid_token_bytes(m["byte_state"], fragment)
                    if m["byte_state"] is None:
                        raise AssertionError("selected ByteLevel token violated UTF-8 state")
                    m["byte_tail"] = (m["byte_tail"] + fragment)[-2:]
                if next_id == eos_id:
                    m["hit_eos"] = True
                    m["done"] = True
                    n_live -= 1
                else:
                    m["answer_ids"].append(next_id)
        for g, i in enumerate(idxs):
            m = meta[i]
            generated_ids = ids_mat[g][len(m["prompt_ids"]):]
            if m["done"]:
                # 出 EOS 之后补的 filler 不算生成内容
                real = len(m["answer_ids"]) + 1
                assert real <= len(generated_ids), "filler trimming exceeded the produced prefix"
                generated_ids = generated_ids[:real]
            out[i] = _finalize_generation(m["row"], m["prompt_ids"], generated_ids,
                                          m["reference_tokens"], m["generation_budget"],
                                          max_new_tokens, m["context_budget"],
                                          m["byte_state"], m["answer_ids"], m["hit_eos"],
                                          m["relaxed"], eos_id, byte_map, tok, utf8_safe)
    missing = [i for i, o in enumerate(out) if o is None]
    assert not missing, "batch generation left rows unprocessed: %s" % missing
    return out


def char_f1(prediction, reference):
    pred, ref = list(normalize_text(prediction)), list(normalize_text(reference))
    if not pred or not ref:
        return float(not pred and not ref)
    common = sum((Counter(pred) & Counter(ref)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall)


def _row_generator(row, temperature, seed, device):
    """每行独立的采样发生器(按 prompt 哈希定种子)⇒ 与行序/前一行长度无关。"""
    if temperature <= 0:
        return None
    prompt_hash = hashlib.blake2b(format_prefix(row).encode("utf-8"), digest_size=8).digest()
    row_seed = (seed + int.from_bytes(prompt_hash, "little")) % (1 << 63)
    return torch.Generator(device=torch.device(device).type).manual_seed(row_seed)


def evaluate_generations(model, tok, rows, max_new_tokens=140, device="cpu", context_limit=96,
                         temperature=0.0, repetition_penalty=1.0, no_repeat_ngram=0,
                         utf8_safe=True, seed=0, batch_min_size=0, q_pool=None):
    """batch_min_size>1: 把 **prompt 等长**且行数 ≥ batch_min_size 的行走批生成路径
    (§51 greedy_generate_batch), 其余仍走逐行路径。默认 0 = 关闭, 行为与优化前完全一致。

    ★§61/§62: `q_pool` 为 (N,d) 或 None。None ⇒ 交给模型自己切(前提 cfg 给了 token id)。
      给了就按行下标取用; 批路径按**组内行序**重排(见 greedy_generate_batch)。
    """
    if not rows:
        raise ValueError("generation evaluation requires at least one QA row")
    # ★ §51: 允许集查表只建一次, 所有行复用(每行重建会把 ~20ms 的 Python BFS 乘上行数)
    eos_id = tok.token_to_id("[EOS]")
    if eos_id is None:
        raise ValueError("tokenizer is missing the required [EOS] token")
    byte_map, _ = bytelevel_token_bytes(tok)
    mask_cache = _DecodeMask(byte_map, eos_id, device) if utf8_safe else None
    outputs = [None] * len(rows)

    def run_single(i):
        return greedy_generate(model, tok, rows[i], max_new_tokens, device,
                               context_limit=context_limit, temperature=temperature,
                               repetition_penalty=repetition_penalty,
                               no_repeat_ngram=no_repeat_ngram, utf8_safe=utf8_safe,
                               generator=_row_generator(rows[i], temperature, seed, device),
                               mask_cache=mask_cache)

    if q_pool is not None:
        assert q_pool.shape[0] == len(rows), \
            "q_pool 的行数(%d)必须等于 rows(%d)" % (q_pool.shape[0], len(rows))

    if batch_min_size and batch_min_size > 1:
        by_len = {}
        for i, row in enumerate(rows):
            plen = len(tok.encode(format_prefix(row), add_special_tokens=False).ids)
            by_len.setdefault(plen, []).append(i)
        leftover = []
        for _plen, idxs in sorted(by_len.items()):
            if len(idxs) < batch_min_size:
                leftover.extend(idxs)
                continue
            gens = [_row_generator(rows[i], temperature, seed, device) for i in idxs]
            # ★ q_pool 按**组内行序**重排(与 greedy_generate_batch 内部一致)
            qp = None if q_pool is None else q_pool[idxs]
            got = greedy_generate_batch(model, tok, [rows[i] for i in idxs], max_new_tokens,
                                        device, context_limit=context_limit,
                                        temperature=temperature,
                                        repetition_penalty=repetition_penalty,
                                        no_repeat_ngram=no_repeat_ngram,
                                        utf8_safe=utf8_safe, mask_cache=mask_cache,
                                        generators=gens, q_pool=qp)
            for j, i in enumerate(idxs):
                outputs[i] = got[j]
    else:
        leftover = list(range(len(rows)))

    # ★§62.5: **所有**没走上批路径的行都走这里, 必须让 q_pool 生效。
    #   第一版把它写成 if/else 的 else 分支 ⇒ batch_min_size>1 时那些"组内行数不够"
    #   的 leftover 行走的是下面那个**不带 q_pool** 的兜底循环, q_pool 静默失效。
    prev = getattr(model, "q_pool", None)
    try:
        for i in leftover:
            if outputs[i] is not None:
                continue
            if q_pool is not None and hasattr(model, "set_question_pool"):
                model.set_question_pool(q_pool[i:i + 1])
            outputs[i] = run_single(i)
    finally:
        if hasattr(model, "set_question_pool"):
            model.set_question_pool(prev)      # 用完还原, 不污染调用方的模型
    scores = []
    for item in outputs:
        p, r = normalize_text(item["generated"]), normalize_text(item["reference"])
        scores.append((float(p == r), char_f1(item["generated"], item["reference"])))
    generation_counts = Counter(normalize_text(item["generated"]) for item in outputs)
    duplicate_generation_records = sum(count for text, count in generation_counts.items()
                                       if text and count > 1)
    return dict(exact_match=sum(x[0] for x in scores) / len(scores),
                char_f1=sum(x[1] for x in scores) / len(scores),
                eos_rate=sum(x["eos"] for x in outputs) / len(outputs),
                length_cap_rate=sum(x["length_cap"] for x in outputs) / len(outputs),
                truncated_context_rate=sum(x["truncated_context"] for x in outputs) / len(outputs),
                incomplete_utf8_rate=sum(x["incomplete_utf8"] for x in outputs) / len(outputs),
                replacement_record_rate=sum(x["replacement_chars"] > 0 for x in outputs) / len(outputs),
                replacement_char_count=sum(x["replacement_chars"] for x in outputs),
                repetition_record_rate=sum(x["repeated_token_count"] > 0 for x in outputs) / len(outputs),
                repeated_token_count=sum(x["repeated_token_count"] for x in outputs),
                mean_repeated_token_rate=sum(x["repeated_token_rate"] for x in outputs) / len(outputs),
                repeated_trigram_count=sum(x["repeated_trigram_count"] for x in outputs),
                mean_repeated_trigram_rate=sum(x["repeated_trigram_rate"] for x in outputs) / len(outputs),
                duplicate_generation_records=duplicate_generation_records,
                duplicate_generation_record_rate=duplicate_generation_records / len(outputs),
                unique_nonempty_generations=sum(bool(text) for text, count in generation_counts.items()),
                constraint_relaxed_steps=sum(x["constraint_relaxed_steps"] for x in outputs),
                reference_longer_than_budget=sum(
                    item["reference_tokens"] > item["generation_budget"] for item in outputs),
                max_reference_tokens=max(item["reference_tokens"] for item in outputs),
                examples=len(outputs)), outputs


def generation_config_from_args(args, sequence_length, saved=None):
    config = dict(max_new_tokens=140, context_limit=sequence_length, temperature=0.0,
                  repetition_penalty=1.0, no_repeat_ngram=0, utf8_safe=True, seed=0)
    if saved:
        config.update({key: saved[key] for key in config if key in saved})
    for key in ("max_new_tokens", "context_limit", "temperature", "repetition_penalty",
                "no_repeat_ngram"):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    if getattr(args, "generation_seed", None) is not None:
        config["seed"] = args.generation_seed
    if getattr(args, "unsafe_utf8", False):
        config["utf8_safe"] = False
    config["max_new_tokens"] = int(config["max_new_tokens"])
    config["context_limit"] = int(config["context_limit"])
    config["temperature"] = float(config["temperature"])
    config["repetition_penalty"] = float(config["repetition_penalty"])
    config["no_repeat_ngram"] = int(config["no_repeat_ngram"])
    config["seed"] = int(config["seed"])
    config["utf8_safe"] = bool(config["utf8_safe"])
    if (config["max_new_tokens"] < 1 or config["max_new_tokens"] > 192 or
            config["context_limit"] < 1 or config["context_limit"] > sequence_length or
            not math.isfinite(config["temperature"]) or config["temperature"] < 0 or
            not math.isfinite(config["repetition_penalty"]) or config["repetition_penalty"] < 1 or
            config["no_repeat_ngram"] < 0):
        raise ValueError("generation settings out of bounds (new tokens 1..192, context 1..training seq)")
    return config


def save_dialogue_checkpoint(path, model, cfg, tokenizer, train_args, data_meta,
                             eval_x, eval_starts, eval_lengths, overwrite=False,
                             generation_rows=None, experiment=None, source_rows=None,
                             source_rows_complete=False):
    tokenizer_meta = dict(kind="bpe", vocab_size=tokenizer.get_vocab_size(), json=tokenizer.to_str())
    save_checkpoint(path, model, cfg, train_args, data_meta, eval_x,
                    overwrite=overwrite, tokenizer=tokenizer_meta,
                    extra=dict(eval_answer_starts=eval_starts.tolist(),
                               eval_lengths=eval_lengths.tolist(),
                               generation_rows=generation_rows or [],
                               raw_dialogue_rows=source_rows or [],
                               raw_dialogue_rows_complete=bool(source_rows_complete),
                               experiment=experiment or {},
                               metric="answer-only teacher-forced and greedy generation"))


def select_generation_rows(train_rows, test_rows, seed=71, n_train=8, n_test=12):
    if len(train_rows) < n_train or len(test_rows) < n_test:
        raise ValueError("not enough encoded records for fixed generation prompt set")
    gen = torch.Generator().manual_seed(seed)
    ti = torch.randperm(len(train_rows), generator=gen)[:n_train].tolist()
    vi = torch.randperm(len(test_rows), generator=gen)[:n_test].tolist()
    return [train_rows[i] for i in ti], [test_rows[i] for i in vi]


@torch.no_grad()
def answer_metrics(model, x, starts, lengths, batch_size, device):
    model.eval()
    total_loss = torch.zeros((), device=device)
    total_tokens = torch.zeros((), dtype=torch.long, device=device)
    correct = torch.zeros((), dtype=torch.long, device=device)
    first_loss = torch.zeros((), device=device)
    first_correct = torch.zeros((), dtype=torch.long, device=device)
    first_count = torch.zeros((), dtype=torch.long, device=device)
    for batch_indices in torch.arange(x.shape[0]).split(batch_size):
        bx = x[batch_indices].to(device)
        starts_b = starts[batch_indices].to(device)
        lengths_b = lengths[batch_indices].to(device)
        positions = torch.arange(bx.shape[1], device=device).view(1, -1)
        valid = positions < lengths_b.view(-1, 1)
        target = bx[:, 1:]
        mask = valid[:, 1:] & (positions[:, 1:] >= starts_b.view(-1, 1))
        logits = model(bx)[:, :-1, :]
        losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                 target.reshape(-1), reduction="none").view_as(target)
        total_loss += (losses * mask).sum()
        total_tokens += mask.sum()
        correct += ((logits.argmax(-1) == target) & mask).sum()
        batch_index = torch.arange(bx.shape[0], device=device)
        first_pos = starts_b - 1
        first_targets = bx[batch_index, starts_b]
        first_logits = logits[batch_index, first_pos]
        first_loss += F.cross_entropy(first_logits, first_targets, reduction="sum")
        first_correct += (first_logits.argmax(-1) == first_targets).sum()
        first_count += bx.shape[0]
    if not bool(total_tokens) or not bool(first_count):
        raise ValueError("no answer tokens in metric split")
    nll = float(total_loss / total_tokens)
    return dict(answer_nll=nll, answer_ppl=math.exp(min(nll, 20)),
                answer_token_accuracy=float(correct / total_tokens),
                answer_tokens=int(total_tokens),
                first_answer_nll=float(first_loss / first_count),
                first_answer_token_accuracy=float(first_correct / first_count),
                first_answer_examples=int(first_count), examples=x.shape[0])


def train_eval_dialogue(model, train_x, train_starts, train_lengths,
                        eval_x, eval_starts, eval_lengths, steps, batch_size,
                        lr, device, seed, first_token_weight=1.0):
    train_dialogue(model, train_x, train_starts, train_lengths, steps, batch_size, lr, device,
                   seed=seed, first_token_weight=first_token_weight)
    return (answer_metrics(model, train_x, train_starts, train_lengths, batch_size, device),
            answer_metrics(model, eval_x, eval_starts, eval_lengths, batch_size, device))


def load_dialogue_checkpoint(path, device):
    model, cfg, payload = train_module.load_checkpoint(path, device)
    tok = Tokenizer.from_str(payload["tokenizer"]["json"])
    if tok.get_vocab_size() != cfg.vocab:
        raise ValueError("serialized BPE vocabulary size differs from RCfg")
    extra = payload.get("extra", {})
    x = payload["eval_tokens"]
    starts = torch.tensor(extra.get("eval_answer_starts", []), dtype=torch.long)
    lengths = torch.tensor(extra.get("eval_lengths", []), dtype=torch.long)
    if x.ndim != 2 or starts.numel() != x.shape[0] or lengths.numel() != x.shape[0]:
        raise ValueError("dialogue checkpoint evaluation tensors/metadata do not align")
    if bool((lengths < 2).any()) or bool((lengths > x.shape[1]).any()):
        raise ValueError("dialogue checkpoint evaluation lengths are out of bounds")
    if bool((starts < 1).any()) or bool((starts >= lengths).any()):
        raise ValueError("dialogue checkpoint answer boundaries are out of bounds")
    pad_id = tok.token_to_id("[PAD]")
    for i, length in enumerate(lengths.tolist()):
        if bool((x[i, length:] != pad_id).any()):
            raise ValueError("dialogue checkpoint padding does not match the serialized tokenizer")
    return model, cfg, tok, x, starts, lengths, payload


def normalize_text(text):
    return "".join(unicodedata.normalize("NFKC", text).casefold().split())


def question_answer_conflicts(rows):
    answers_by_prompt = {}
    for row in rows:
        prompt = normalize_text(format_prefix(row))
        answers_by_prompt.setdefault(prompt, set()).add(normalize_text(row["answer"]))
    conflicting = {prompt: answers for prompt, answers in answers_by_prompt.items()
                   if len(answers) > 1}
    return dict(conflicting_prompts=len(conflicting),
                examples=[dict(prompt=prompt, answers=sorted(answers))
                          for prompt, answers in list(conflicting.items())[:8]])


def split_records(rows, train_fraction=0.8, seed=29, min_groups=3, return_meta=False):
    """Keep rows connected by a normalized shared question OR answer in one split."""
    if len(rows) < 4 or not 0.5 <= train_fraction < 1.0 or min_groups < 3:
        raise ValueError("question/answer-disjoint split needs >=4 rows, >=3 groups and a valid fraction")
    parent = list(range(len(rows)))

    def find(i):
        for _ in range(len(parent)):
            if parent[i] == i:
                return i
            parent[i] = parent[parent[i]]
            i = parent[i]
        raise AssertionError("question/answer component parent chain exceeded row count")

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    seen_q, seen_a = {}, {}
    for i, row in enumerate(rows):
        q, a = normalize_text(row["question"]), normalize_text(row["answer"])
        if q in seen_q:
            union(i, seen_q[q])
        else:
            seen_q[q] = i
        if a in seen_a:
            union(i, seen_a[a])
        else:
            seen_a[a] = i
    components = {}
    for i in range(len(rows)):
        components.setdefault(find(i), []).append(i)
    groups = list(components.values())
    if len(groups) < min_groups:
        raise ValueError("only %d disconnected groups; need at least %d" % (len(groups), min_groups))
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(groups), generator=g).tolist()
    target = int(len(rows) * (1.0 - train_fraction))
    eval_group_ids, eval_count = set(), 0
    for gi in order:
        if len(eval_group_ids) >= len(groups) - 1:
            break
        eval_group_ids.add(gi)
        eval_count += len(groups[gi])
        if eval_count >= target:
            break
    held_indices = {i for gi in eval_group_ids for i in groups[gi]}
    train = [row for i, row in enumerate(rows) if i not in held_indices]
    heldout = [row for i, row in enumerate(rows) if i in held_indices]
    train_q = {normalize_text(r["question"]) for r in train}
    eval_q = {normalize_text(r["question"]) for r in heldout}
    train_a = {normalize_text(r["answer"]) for r in train}
    eval_a = {normalize_text(r["answer"]) for r in heldout}
    if not train or not heldout or train_q & eval_q or train_a & eval_a:
        raise AssertionError("normalized question/answer leakage or empty split")
    meta = dict(components=len(groups), train_components=len(groups) - len(eval_group_ids),
                eval_components=len(eval_group_ids), train_questions=len(train_q),
                eval_questions=len(eval_q), train_answers=len(train_a), eval_answers=len(eval_a),
                question_overlap=len(train_q & eval_q), answer_overlap=len(train_a & eval_a),
                train_component_sizes=[len(groups[i]) for i in range(len(groups)) if i not in eval_group_ids],
                eval_component_sizes=[len(groups[i]) for i in range(len(groups)) if i in eval_group_ids])
    return (train, heldout, meta) if return_meta else (train, heldout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load_checkpoint")
    ap.add_argument("--decoder_ablation", action="store_true",
                    help="对同一固定 checkpoint 运行原始/UTF-8/重复惩罚/温度解码小对照")
    ap.add_argument("--audit_checkpoint_text", action="store_true",
                    help="只审计 checkpoint 中序列化的样本和 tokenizer 往返，不训练、不覆盖 checkpoint")
    ap.add_argument("--learning_test", action="store_true",
                    help="3 seeds AdamW 学习/答案与问题双留出 greedy QA 测试")
    ap.add_argument("--report_jsonl", default="dialogue_qa_answer_disjoint_smoke_v5.jsonl")
    ap.add_argument("--max_new_tokens", type=int)
    ap.add_argument("--context_limit", type=int)
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--repetition_penalty", type=float)
    ap.add_argument("--no_repeat_ngram", type=int)
    ap.add_argument("--generation_seed", type=int)
    ap.add_argument("--unsafe_utf8", action="store_true")
    ap.add_argument("--prepare_only", action="store_true",
                    help="只读受限数据样本、切分并训练 BPE；不建模型、不训练、不写 checkpoint")
    ap.add_argument("--save_checkpoint", default="engram_rule_dialogue_bpe_answer_disjoint_learning_v4.pt")
    ap.add_argument("--overwrite_checkpoint", action="store_true")
    ap.add_argument("--max_bytes", type=int, default=16_777_216)
    ap.add_argument("--max_lines", type=int, default=4096)
    ap.add_argument("--rows_per_range", type=int, default=1024)
    ap.add_argument("--vocab_size", type=int, default=512)
    ap.add_argument("--max_len", type=int)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--train_seed", type=int, default=7)
    ap.add_argument("--first_token_weight", type=float, default=1.0)
    a = ap.parse_args()
    if a.load_checkpoint and (a.prepare_only or a.learning_test):
        ap.error("--load_checkpoint 不能与 --prepare_only/--learning_test 同时使用")
    if a.prepare_only and a.overwrite_checkpoint:
        ap.error("--prepare_only 不写 checkpoint，不能使用 --overwrite_checkpoint")
    if a.learning_test and a.overwrite_checkpoint:
        ap.error("learning_test 使用新版本 checkpoint 路径，不自动覆盖")
    if (a.decoder_ablation or a.audit_checkpoint_text) and not a.load_checkpoint:
        ap.error("--decoder_ablation/--audit_checkpoint_text requires --load_checkpoint")
    if a.audit_checkpoint_text and a.decoder_ablation:
        ap.error("--audit_checkpoint_text and --decoder_ablation are separate read-only operations")
    if a.load_checkpoint:
        model, cfg, tok, eval_x, eval_starts, eval_lengths, payload = load_dialogue_checkpoint(a.load_checkpoint, DEV)
        experiment = payload.get("extra", {}).get("experiment", {})
        if a.audit_checkpoint_text:
            audit_rows = payload.get("extra", {}).get("raw_dialogue_rows", [])
            audit_scope = "saved_sample"
            if not audit_rows:
                prompts = experiment.get("prompts", {})
                if isinstance(prompts, dict):
                    for split in ("train", "eval"):
                        audit_rows.extend(prompts.get(split, []))
                audit_scope = "generation_prompts_only"
            if not audit_rows:
                audit_rows = payload.get("extra", {}).get("generation_rows", [])
                audit_scope = "generation_rows_only"
            if not audit_rows:
                raise ValueError("checkpoint contains no serialized QA rows to audit")
            result = audit_text_rows(audit_rows, tok)
            result.update(checkpoint=os.path.abspath(a.load_checkpoint),
                          checkpoint_kind=payload.get("corpus", {}).get("kind"),
                          audit_scope=audit_scope,
                          saved_rows_complete=bool(payload.get("extra", {}).get("raw_dialogue_rows_complete", False)),
                          training_rows_count=payload.get("corpus", {}).get("raw_train_rows"),
                          prompt_answer_conflicts=question_answer_conflicts(audit_rows))
            print(json.dumps(result, ensure_ascii=False))
            return 0 if result["clean"] else 2
        training_seq = int(payload["training"]["seq"])
        experiment = payload.get("extra", {}).get("experiment", {})
        saved_generation = experiment.get("generation", {})
        generation = generation_config_from_args(a, training_seq, saved_generation)
        if a.report_jsonl is None:
            base, _ = os.path.splitext(a.load_checkpoint)
            a.report_jsonl = base + "_decoder.jsonl"
        if os.path.exists(a.report_jsonl):
            raise FileExistsError("refusing to overwrite decoder report: %s" % a.report_jsonl)
        report = evaluate_dialogue(model, eval_x, eval_starts, eval_lengths, a.bs, DEV)
        print("loaded tokenizer=BPE vocab=%d | data=%s" % (cfg.vocab, payload["corpus"]))
        print("answer-only metrics", report)
        prompts = experiment.get("prompts", {})
        rows = prompts.get("eval", payload.get("extra", {}).get("generation_rows", []))
        if rows:
            variants = [dict(name="configured", **generation)]
            if a.decoder_ablation:
                variants = [
                    dict(name="legacy_unfiltered_cap32", **dict(generation, max_new_tokens=32,
                         utf8_safe=False, repetition_penalty=1.0, no_repeat_ngram=0, temperature=0.0)),
                    dict(name="legacy_unfiltered_cap140", **dict(generation, max_new_tokens=140,
                         utf8_safe=False, repetition_penalty=1.0, no_repeat_ngram=0, temperature=0.0)),
                    dict(name="utf8_safe_greedy", **dict(generation, max_new_tokens=140, utf8_safe=True,
                         repetition_penalty=1.0, no_repeat_ngram=0, temperature=0.0)),
                    dict(name="utf8_plus_repeat", **dict(generation, utf8_safe=True,
                         repetition_penalty=1.2, no_repeat_ngram=3, temperature=0.0)),
                    dict(name="utf8_temperature", **dict(generation, utf8_safe=True,
                         repetition_penalty=1.0, no_repeat_ngram=0, temperature=0.7)),
                ]
            with open(a.report_jsonl, "x", encoding="utf-8") as f:
                f.write(json.dumps(dict(type="reload_evaluation", answer_teacher_forced=report,
                                        checkpoint=a.load_checkpoint, generation=generation,
                                        decoder_ablation=a.decoder_ablation), ensure_ascii=False) + "\n")
                for variant in variants:
                    settings = {k: v for k, v in variant.items() if k != "name"}
                    metrics, outputs = evaluate_generations(model, tok, rows, device=DEV, **settings)
                    print(variant["name"], metrics)
                    f.write(json.dumps(dict(type="decoder_metrics", variant=variant["name"],
                                            metrics=metrics), ensure_ascii=False) + "\n")
                    for item in outputs:
                        f.write(json.dumps(dict(type="reload_generation", variant=variant["name"],
                                                split="eval", **item), ensure_ascii=False) + "\n")
            print("saved decoder report", os.path.abspath(a.report_jsonl))
        else:
            print("checkpoint has no saved generation prompts; decoder evaluation skipped")
        return 0
    learning = a.learning_test
    if a.report_jsonl is None:
        a.report_jsonl = "dialogue_qa_learning_v4.jsonl"
    if a.max_bytes > 16_777_216 or a.max_lines > 4096 or a.rows_per_range > 1024:
        raise ValueError("bounded-test caps exceeded (16MiB, 4096 rows, 1024 rows per range)")
    if a.vocab_size < 262 or a.vocab_size > 2048 or (a.max_len is not None and
            (a.max_len < 4 or a.max_len > 1024)):
        raise ValueError("bounded-test BPE/sequence bounds: vocab 262..2048, seq 4..1024")
    step_cap, batch_floor, batch_cap = (60, 256, 256) if learning else (200, 1, 256)
    if a.steps < 1 or a.steps > step_cap or a.bs < batch_floor or a.bs > batch_cap or a.lr <= 0:
        raise ValueError("training bounds: steps 1..%d, batch %d..%d, lr > 0" %
                         (step_cap, batch_floor, batch_cap))
    if a.decoder_ablation or a.unsafe_utf8:
        raise ValueError("decoder_ablation/unsafe_utf8 apply only when loading a saved checkpoint")
    if learning and not a.prepare_only:
        if (a.seeds != 3 or a.steps != 60 or a.bs != 256 or
                "answer_disjoint" not in os.path.basename(a.save_checkpoint) or
                not a.save_checkpoint.endswith("learning_v4.pt")):
            raise ValueError("learning_test fixes 3 seeds, 60 steps, batch 256, and a distinct answer_disjoint *_learning_v4.pt checkpoint")
        if os.path.exists(a.save_checkpoint):
            raise FileExistsError("refusing to overwrite learning checkpoint: %s" % a.save_checkpoint)
        if os.path.exists(a.report_jsonl):
            raise FileExistsError("refusing to overwrite learning report: %s" % a.report_jsonl)

    rows, source_meta = sample_ranges(max_bytes=a.max_bytes, max_lines=a.max_lines,
                                     rows_per_range=a.rows_per_range)
    source_meta["sample_text_audit"] = require_clean_text_rows(rows)
    group_meta = None
    if learning:
        train_rows, test_rows, group_meta = split_records(rows, seed=29, min_groups=20, return_meta=True)
    else:
        train_rows, test_rows = split_records(rows)
    train_conflicts = question_answer_conflicts(train_rows)
    eval_conflicts = question_answer_conflicts(test_rows)
    source_meta["prompt_answer_conflicts"] = dict(train=train_conflicts, eval=eval_conflicts)
    if len(train_rows) < (32 if learning else 16) or len(test_rows) < (12 if learning else 4):
        raise ValueError("train/eval split left too few QA rows")
    tok, bpe_audit = build_split_bpe(train_rows, test_rows, a.vocab_size)
    source_meta["bpe_audit"] = bpe_audit
    required_len, length_stats = required_sequence_length(train_rows, tok, percentile=1.0)
    selected_length = max(192, required_len) if a.max_len is None else a.max_len
    if required_len > selected_length:
        raise ValueError("complete training QA sequence requires %d tokens; --max_len is %d" %
                         (required_len, selected_length))
    if selected_length > 1024:
        raise ValueError("training QA sequences exceed the bounded-test 1024-token limit")
    a.max_len = selected_length
    source_meta["sequence_sizing"] = dict(**length_stats, selected_length=a.max_len,
                                           policy="max derived from train split only; no left truncation")
    train_x, train_starts, train_lengths, train_kept_rows = encode_examples(
        train_rows, tok, a.max_len, return_rows=True)
    if len(train_kept_rows) != len(train_rows):
        raise AssertionError("train-derived context sizing excluded a training record")
    eval_rows_before_context_filter = len(test_rows)
    test_rows, eval_context_filter = partition_by_token_budget(test_rows, tok, a.max_len)
    if not test_rows:
        raise ValueError("no complete heldout QA rows fit the training-derived context window")
    eval_x, eval_starts, eval_lengths, eval_kept_rows = encode_examples(
        test_rows, tok, a.max_len, return_rows=True)
    if len(eval_kept_rows) != len(test_rows):
        raise AssertionError("heldout token-budget partition disagrees with encoding")
    source_meta["eval_context_filter"] = dict(rows_before=eval_rows_before_context_filter,
                                               rows_after=len(eval_kept_rows), **eval_context_filter,
                                               policy="complete rows only; no left truncation")
    train_rows, test_rows = train_kept_rows, eval_kept_rows
    generation = generation_config_from_args(a, a.max_len)
    if learning:
        gen_train_rows, gen_test_rows = select_generation_rows(train_rows, test_rows)
    else:
        gen_train_rows = gen_test_rows = []
    if train_x.shape[0] < (32 if learning else 16) or eval_x.shape[0] < (12 if learning else 4):
        raise ValueError("too few encoded train/eval examples after answer-boundary and length filtering")
    if train_x.shape[0] + eval_x.shape[0] > a.max_lines:
        raise AssertionError("post-tokenization sample count exceeded configured cap")
    if tok.get_vocab_size() > a.vocab_size or tok.token_to_id("[PAD]") is None:
        raise AssertionError("trained BPE violates configured vocab/special-token contract")
    print("prepared bytes=%d rows=%d encoded train=%d eval=%d groups=%s history_rows=%d vocab=%d seq=%d"
          % (source_meta["bytes_read"], source_meta["rows"], train_x.shape[0], eval_x.shape[0],
             group_meta, source_meta["multi_turn_rows"], tok.get_vocab_size(), a.max_len))
    if a.prepare_only:
        print("prepare-only: no model initialization, training, or checkpoint write")
        return 0
    if not learning and os.path.exists(a.save_checkpoint) and not a.overwrite_checkpoint:
        raise FileExistsError("checkpoint exists; refusing overwrite: %s" % a.save_checkpoint)
    if not learning and os.path.exists(a.report_jsonl):
        raise FileExistsError("refusing to overwrite dialogue report: %s" % a.report_jsonl)

    cfg = RCfg(vocab=tok.get_vocab_size(), d=64, n_layers=2, n_heads=4, ffn_mult=2, rope_dim=8,
               ngram_max=2, n_hash=2, n_slots=1 << 12, min_count=1, n_buckets=16,
               gate_rank=16, pos_abs_max=max(256, a.max_len), use_pos_bucket=True, window_max=16,
               bucket_step=max(1, -(-a.max_len // 16)))
    corpus_meta = dict(kind="modelscope_qiaojiedongfeng_learning_test" if learning else
                       "modelscope_qiaojiedongfeng_dialogue_small",
                       dataset_id="qiaojiedongfeng/qiaojiedongfeng", revision=DATA_REVISION,
                       license=DATA_LICENSE, license_scope="dataset metadata declaration; file-level rights not independently verified",
                       source_disclosed=False, sample=source_meta, split_seed=29,
                       split_group="normalized_question_or_answer_connected_component",
                       group_split=group_meta, raw_train_rows=len(train_rows), raw_eval_rows=len(test_rows),
                       encoded_train_rows=int(train_x.shape[0]), encoded_eval_rows=int(eval_x.shape[0]),
                       history_rows=source_meta["multi_turn_rows"],
                       scope="bounded single-turn question/answer learning and generation test; not a multi-turn benchmark")

    if not learning:
        torch.manual_seed(a.train_seed)
        model = EngramRuleLM(cfg).to(DEV)
        build_engram_from_examples(model, train_x, train_lengths)
        before = answer_metrics(model, train_x, train_starts, train_lengths, a.bs, DEV)
        before_eval = answer_metrics(model, eval_x, eval_starts, eval_lengths, a.bs, DEV)
        train_dialogue(model, train_x, train_starts, train_lengths, a.steps, a.bs, a.lr, DEV,
                       seed=a.train_seed, first_token_weight=a.first_token_weight)
        train_report = answer_metrics(model, train_x, train_starts, train_lengths, a.bs, DEV)
        eval_report = answer_metrics(model, eval_x, eval_starts, eval_lengths, a.bs, DEV)
        eval_generation_rows = select_generation_rows(train_rows, test_rows, n_train=8, n_test=12)[1]
        generation_metrics, generation_outputs = evaluate_generations(
            model, tok, eval_generation_rows, device=DEV, **generation)
        report = dict(train_seed=a.train_seed, seed_count=1, seed_range_unavailable=True,
                      training_config=dict(steps=a.steps, batch_size=a.bs, lr=a.lr,
                                           seq=a.max_len, vocab=cfg.vocab,
                                           first_token_weight=a.first_token_weight),
                      data_split=dict(sample_sha256=source_meta["sample_sha256"],
                                      split_seed=29, train_rows=len(train_rows),
                                      eval_rows=len(test_rows),
                                      train_tokens=int((train_lengths - train_starts).sum()),
                                      eval_tokens=int((eval_lengths - eval_starts).sum())),
                      first_token_weight=a.first_token_weight,
                      train_before=before, eval_before=before_eval,
                      train_after=train_report, eval_after=eval_report,
                      heldout_generation=generation_metrics)
        with open(a.report_jsonl, "x", encoding="utf-8") as f:
            f.write(json.dumps(dict(type="summary", data=report), ensure_ascii=False) + "\n")
            for item in generation_outputs:
                f.write(json.dumps(dict(type="generation", split="eval", **item),
                                   ensure_ascii=False) + "\n")
        save_dialogue_checkpoint(a.save_checkpoint, model, cfg, tok,
                                 dict(steps=a.steps, bs=a.bs, seq=a.max_len, lr=a.lr,
                                      train_seed=a.train_seed, eval_seed=0), corpus_meta,
                                 eval_x, eval_starts, eval_lengths, a.overwrite_checkpoint,
                                 generation_rows=eval_generation_rows,
                                 experiment=dict(generation=generation,
                                                 generation_metrics=generation_metrics,
                                                 first_token_weight=a.first_token_weight,
                                                 prompts=dict(eval=eval_generation_rows)),
                                 source_rows=rows, source_rows_complete=True)
        print("answer metrics", report)
        print("heldout prompt-only generation", generation_metrics)
        print("saved checkpoint", os.path.abspath(a.save_checkpoint))
        print("saved report", os.path.abspath(a.report_jsonl))
        return 0

    gen_payload = dict(train=[{k: row[k] for k in ("question", "answer", "history")} for row in gen_train_rows],
                       eval=[{k: row[k] for k in ("question", "answer", "history")} for row in gen_test_rows])
    results, seed_outputs, seed0_model = [], [], None
    for seed in (11, 23, 37):
        torch.manual_seed(seed)
        model = EngramRuleLM(cfg).to(DEV)
        build_engram_from_examples(model, train_x, train_lengths)
        before_train = answer_metrics(model, train_x, train_starts, train_lengths, 256, DEV)
        before_eval = answer_metrics(model, eval_x, eval_starts, eval_lengths, 256, DEV)
        pre_train_gen, _ = evaluate_generations(model, tok, gen_train_rows, device=DEV, **generation)
        pre_eval_gen, _ = evaluate_generations(model, tok, gen_test_rows, device=DEV, **generation)
        after_train, after_eval = train_eval_dialogue(
            model, train_x, train_starts, train_lengths, eval_x, eval_starts, eval_lengths,
            a.steps, 256, a.lr, DEV, seed, a.first_token_weight)
        train_gen, train_rows_out = evaluate_generations(model, tok, gen_train_rows, device=DEV, **generation)
        eval_gen, eval_rows_out = evaluate_generations(model, tok, gen_test_rows, device=DEV, **generation)
        results.append(dict(seed=seed, before_train=before_train, before_eval=before_eval,
                            after_train=after_train, after_eval=after_eval,
                            before_train_generation=pre_train_gen, before_eval_generation=pre_eval_gen,
                            after_train_generation=train_gen, after_eval_generation=eval_gen))
        seed_outputs.append(dict(seed=seed, train=train_rows_out, eval=eval_rows_out))
        print("seed %d train NLL %.4f→%.4f EM %.3f→%.3f | holdout NLL %.4f→%.4f EM %.3f→%.3f charF1 %.3f"
              % (seed, before_train["answer_nll"], after_train["answer_nll"],
                 pre_train_gen["exact_match"], train_gen["exact_match"],
                 before_eval["answer_nll"], after_eval["answer_nll"],
                 pre_eval_gen["exact_match"], eval_gen["exact_match"], eval_gen["char_f1"]))
        if seed == 11:
            seed0_model = model
        else:
            del model

    def summarize(path):
        vals = [r[path[0]][path[1]] for r in results]
        return dict(mean=sum(vals) / len(vals), min=min(vals), max=max(vals), range=max(vals) - min(vals))

    summary = dict(task="bounded_single_turn_qa_not_multi_turn", group_split=group_meta,
                   training_config=dict(steps=a.steps, batch_size=256, lr=a.lr,
                                        seeds=[11, 23, 37], seq=a.max_len, vocab=cfg.vocab,
                                        first_token_weight=a.first_token_weight),
                   data_split=dict(sample_sha256=source_meta["sample_sha256"],
                                   train_rows=len(train_rows), eval_rows=len(test_rows),
                                   train_answer_tokens=int((train_lengths - train_starts).sum()),
                                   eval_answer_tokens=int((eval_lengths - eval_starts).sum())),
                   seeds=results,
                   train_nll_before=summarize(("before_train", "answer_nll")),
                   train_nll_after=summarize(("after_train", "answer_nll")),
                   eval_nll_before=summarize(("before_eval", "answer_nll")),
                   eval_nll_after=summarize(("after_eval", "answer_nll")),
                   train_em_before=summarize(("before_train_generation", "exact_match")),
                   train_em_after=summarize(("after_train_generation", "exact_match")),
                   eval_em_before=summarize(("before_eval_generation", "exact_match")),
                   eval_em_after=summarize(("after_eval_generation", "exact_match")),
                   eval_char_f1_after=summarize(("after_eval_generation", "char_f1")),
                   decoder_config=generation,
                   notes="N=3 seed range is descriptive, not a significance test; no history rows sampled")
    os.makedirs(os.path.dirname(os.path.abspath(a.report_jsonl)), exist_ok=True)
    with open(a.report_jsonl, "w", encoding="utf-8") as f:
        f.write(json.dumps(dict(type="summary", data=summary), ensure_ascii=False) + "\n")
        for seed_result in seed_outputs:
            for split in ("train", "eval"):
                for item in seed_result[split]:
                    f.write(json.dumps(dict(type="generation", seed=seed_result["seed"],
                                            split=split, **item), ensure_ascii=False) + "\n")
    save_dialogue_checkpoint(a.save_checkpoint, seed0_model, cfg, tok,
                              dict(steps=60, bs=256, seq=a.max_len, lr=a.lr,
                                   first_token_weight=a.first_token_weight,
                                  train_seed=11, eval_seed=0, seeds=[11, 23, 37]), corpus_meta,
                             eval_x, eval_starts, eval_lengths,
                             generation_rows=gen_payload["eval"],
                             experiment=dict(summary=summary, prompts=gen_payload,
                                             generation=generation,
                                              first_token_weight=a.first_token_weight,
                                             generation_limit=generation["max_new_tokens"],
                                             train_generation_count=len(gen_payload["train"]),
                                             eval_generation_count=len(gen_payload["eval"])),
                              source_rows=rows, source_rows_complete=True)

    print("saved seed-11 checkpoint", os.path.abspath(a.save_checkpoint))
    print("saved generation report", os.path.abspath(a.report_jsonl))
    print("learning/generalization summary", json.dumps(summary, ensure_ascii=False))
    print("Conclusion gate: train-fit requires train NLL down and train generation up; generalization requires heldout generation metrics plus group isolation and per-seed range.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
