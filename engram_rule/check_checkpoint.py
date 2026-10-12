"""Checkpoint round-trip + refusal-to-overwrite checks.

Run: python engram_rule/check_checkpoint.py
"""
import os
import tempfile

import torch

from model import EngramMemory, EngramRuleLM, RCfg
import train as train_module
from train import load_checkpoint, save_checkpoint


def main():
    cfg = RCfg(vocab=32, d=32, n_layers=1, n_heads=4, ffn_mult=2, rope_dim=8,
               ngram_max=2, n_hash=2, n_slots=64, min_count=1, n_buckets=8,
               gate_rank=8, pos_abs_max=32, window_max=4, bucket_step=1)
    model = EngramRuleLM(cfg).eval()
    with torch.no_grad():
        model.engram.alive[:7] = True
    x = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
    with torch.no_grad():
        expected = model(x)

    train_meta = dict(steps=1, bs=2, seq=8, lr=1e-3, train_seed=7, eval_seed=7)
    corpus_meta = dict(kind="synthetic_zipf", vocab_mapping=None, tokenizer=None,
                       tokens=32, alpha=1.1, seed=0, train_fraction=0.8,
                       engram_build_tokens=16)
    with tempfile.TemporaryDirectory(prefix="engram-rule-checkpoint-") as td:
        path = os.path.join(td, "roundtrip.pt")
        eval_tokens = torch.arange(1, 25, dtype=torch.long).remainder(cfg.vocab)
        save_checkpoint(path, model, cfg, train_meta, corpus_meta, eval_tokens)
        loaded, loaded_cfg, payload = load_checkpoint(path, "cpu")
        with torch.no_grad():
            actual = loaded(x)
        assert loaded_cfg.__dict__ == cfg.__dict__
        assert payload["tokenizer"] is None
        assert payload["corpus"]["kind"] == "synthetic_zipf"
        assert torch.equal(payload["eval_tokens"], eval_tokens)
        assert set(loaded.state_dict()) == set(model.state_dict())
        assert all(torch.equal(model.state_dict()[k], loaded.state_dict()[k])
                   for k in model.state_dict())
        assert torch.equal(expected, actual), "logits changed across checkpoint round-trip"
        assert torch.equal(model.engram.alive, loaded.engram.alive)

        original_zipf = train_module.zipf_data
        original_train = train_module.train_model
        original_build = EngramMemory.build_from_corpus
        def forbidden(*args, **kwargs):
            raise AssertionError("checkpoint load path unexpectedly trained or rebuilt Engram")
        train_module.zipf_data = forbidden
        train_module.train_model = forbidden
        EngramMemory.build_from_corpus = forbidden
        try:
            from unittest.mock import patch
            with patch("sys.argv", ["train.py", "--load_checkpoint", path]):
                assert train_module.main() == 0
        finally:
            train_module.zipf_data = original_zipf
            train_module.train_model = original_train
            EngramMemory.build_from_corpus = original_build

        bad = dict(payload)
        bad["training"] = dict(payload["training"])
        del bad["training"]["eval_seed"]
        bad_path = os.path.join(td, "missing-metadata.pt")
        torch.save(bad, bad_path)
        try:
            load_checkpoint(bad_path, "cpu")
        except ValueError:
            pass
        else:
            raise AssertionError("checkpoint with missing reproducibility metadata was accepted")

        protected = os.path.join(td, "protected.pt")
        with open(protected, "wb") as f:
            f.write(b"do-not-overwrite")
        try:
            save_checkpoint(protected, model, cfg, train_meta, corpus_meta, eval_tokens)
        except FileExistsError:
            pass
        else:
            raise AssertionError("existing checkpoint was overwritten without explicit opt-in")
        with open(protected, "rb") as f:
            assert f.read() == b"do-not-overwrite"

    print("PASS checkpoint config/state/buffers/logits round-trip; existing file remains untouched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
