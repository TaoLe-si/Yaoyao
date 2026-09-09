# Shuffled checkpoint compatibility: conditional CPU-only infrastructure

New files only. No trainer, live checkpoint, CUDA source, pilot process, or production integration is changed. Parent builds/tests; this subagent did not compile or run tests. Shell pwd could not launch (tool reported File not found); explicit D:/TaoVm paths were used. These are scheduling metadata tests, not GPU resumability proof.

## Existing behavior and adapter sequence

Reviewed shuffled_epoch_cursor.hpp, slot_checkpoint_file.cuh, slot_resume_snapshot.cuh and CPU cursor dependencies. SCP1 checks doc < next and target <= data.docs[doc].size(). ShuffledEpochCursor uses canonical doc IDs but next is permutation allocation position. Changing only the saved doc field is insufficient unless the loader receives physical-order documents.

1. At a successfully committed step boundary, capture the canonical SlotSnapshot and shuffled scheduling sidecar from the same state. No pending work may be checkpointed.
2. Construct ShuffledCheckpointMapping(canonical, shuffled.order()). Use before_save(snapshot.cursor, snapshot.next) and assign the result only into a COPY of snapshot. next, targets, empty sentinels and slot ordering stay unchanged; doc becomes inverse-permutation rank. Never reorder s/mem or any model/optimizer tensors. Pass the copy to existing save_slot_file.
3. On future load, verify the proposed outer transaction described below first, restore the canonical shuffled sidecar, and construct the mapping from its order and verified canonical documents.
4. Use mapping.load_template(canonical) as the data parameter to EXISTING load_slot_file. This is a copied, already-normalized PilotCursor with docs[rank] = canonical.docs[order[rank]], empty slots and next=0. It does not reconstruct via the legacy EOS-stripping constructor. Existing expected SlotSnapshot still supplies tensor shapes.
5. Call after_load(loaded.cursor, loaded.next, restored_sidecar.cursor()). It converts rank back to canonical ID and rejects any next/slot/doc/target disagreement, invalid ranges, duplicate active docs or bad empty sentinel. Assign decoded metadata to the loaded snapshot ONLY after success. Keep the canonical runtime docs and restored ShuffledEpochCursor for future allocation, not the physical template's ordinary take(). Existing ShuffledEpochCursor::restore itself replaces input slots from the sidecar, so explicit cross-check is essential.
6. Only then may future production integration invoke tensor restore. No such integration is supplied here.

The helper verifies mapping/length/metadata consistency, not corpus token bytes, provenance, seed-derived permutation authenticity, tensor states or checkpoint file bytes. It is not a serializer or cryptographic layer. It copies document storage for the load template; production memory cost must be reviewed later.

## Future outer manifest (proposal only, not implemented)

Publish a versioned, unambiguous canonical manifest as the last atomic commit marker in a unique checkpoint transaction directory after writing/closing all artifacts. Bind:

- exact full SCP file SHA-256 and byte count (including SCP1 header/checksum), checkpoint format/version and explicit cursor encoding permutation-rank-v1;
- exact canonical shuffled sidecar SHA-256 and byte count;
- permutation algorithm/version, canonical order digest (SHA-256 of domain-separated fixed-width little-endian count + rank-to-canonical IDs), epoch, seed and shuffle flag;
- verified corpus/tokenizer/preprocessing content identities, canonical document count/order, slot count, step and transaction ID;
- any accompanying model/control artifact hashes needed by the complete checkpoint transaction.

Load fails closed for a missing/partial manifest, altered files, incompatible algorithm/encoding, or inconsistent sidecar epoch/seed/order. Recompute all hashes from actual bytes and compare decoded sidecar fields and permutation digest; recompute the deterministic order for that epoch/seed/algorithm where applicable (legacy unshuffled epochs use identity order). Apply explicit size bounds before parsing/loading. Commit strategy and crash/durability handling need platform review.

Existing SCP bundle_hash is not an authenticated security boundary. Neither an identity-shaped string nor a checkpoint_id accepted by ShuffledEpochCursor save/restore verifies its contents. SHA-256 plus a local manifest detects accidental mismatch under a trusted manifest; authenticity against replacement of all files requires a separately trusted manifest digest, signature or MAC. Do not claim secure binding until that verification/publishing layer is implemented and tested.

## CPU tests and parent build

C++17, standard library only, no .cuh includes or CUDA linkage. For example, compile test_shuffled_checkpoint_metadata.cpp with cl /std:c++17 /EHsc or g++ -std=c++17. Parent chooses a non-live build output location.

Tests use unequal lengths 3/11/5/8/4 and explicitly demonstrate canonical-doc loader rejection of a valid rank-encoded long-document target, then acceptance with the physical template. They cover bijection/range/shape/target/sentinel/duplicate/sidecar mismatch rejection, canonical doc >= next with rank < next, legacy identity mapping, and interrupted target-by-target scheduling traces at every call boundary for three seeds and slot counts 1/3/7, including epoch rollover and repeated next-epoch interruptions. Tests model ONLY existing loader cursor checks; they do not run the real SCP disk/tensor loader. Assertions remain active in release builds.

Exact new files:
- D:/TaoVm/shuffled_checkpoint_metadata.hpp
- D:/TaoVm/test_shuffled_checkpoint_metadata.cpp
- D:/TaoVm/SHUFFLED_CHECKPOINT_METADATA_PROPOSAL.md
