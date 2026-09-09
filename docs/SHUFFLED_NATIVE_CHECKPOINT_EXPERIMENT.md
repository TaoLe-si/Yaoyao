# Experimental native shuffled checkpoint wrapper

New header: shuffled_native_checkpoint.cuh. New documentation: this file. No existing code or live artifacts changed; no GPU commands, compilation or execution performed by this subagent. Parent review/build is required. Earlier CPU metadata tests passing does NOT establish native checkpoint or GPU resume equivalence.

## API (namespace tao::dual::shuffled_native)

- save_snapshot(new_directory, captured_snapshot, shuffled_epoch, seed, verified_identity, scp_identity, limits): CPU serialization adapter around existing save_slot_file. Copies the snapshot, maps only cursor.doc to permutation rank, checks canonical sidecar agreement, and leaves tensors/slot order untouched.
- save(new_directory, sequence_slots, shuffled_epoch, seed, verified_identity, scp_identity, limits): native convenience function. Copies canonical PilotCursor to satisfy existing mutable capture signature; capture copies existing tensor states, not their ordering. This function uses existing device transfers if called; none were called here.
- load_snapshot(directory, trusted_manifest_sha256, expected_snapshot, verified_canonical_cursor, expected_seed, verified_identity, scp_identity, limits): validates transaction and returns Loaded{snapshot, epoch}. Snapshot metadata is canonical again. Existing load_slot_file receives a physical-order document template so its docs[rank] length checks are correct. No tensor mutation in this function.
- load_and_restore(directory, trusted_manifest_sha256, sequence_slots, epoch_reference, expected_seed, verified_identity, scp_identity, limits): captures expected tensor shapes at a safe current boundary, calls load_snapshot, then existing tensor restore, then adopts the restored ShuffledEpochCursor. Do not call during pending work or training. Real GPU resume validation remains deferred.

The save result is Commit{directory, manifest_sha256}. Preserve that digest in a separately trusted receipt. Loading requires it; do not manufacture the expected digest by hashing the candidate manifest at load time. Identity strings must refer to externally verified canonical corpus/tokenizer/preprocessing bytes; the wrapper hashes those strings for binding, but cannot establish their truth. scp_identity remains the existing checkpoint_identity contract.

## Transaction and verification

Caller chooses an unused, private, non-live directory path. Absolute and ordinary relative paths are accepted; drive-relative Windows paths, dot/traversal/control-byte components and symlink ancestors are rejected. The parent must exist. Atomic create_directory reserves a unique transaction name; existing/stale directories are refused, so no artifact is overwritten by cooperating callers. Artifact names are fixed and manifest-validated: state.scp, epoch.sidecar and manifest. A manifest cannot redirect the loader to arbitrary absolute or relative artifact paths.

Existing serializer publishes state.scp via its temporary file/rename. Sidecar is written to epoch.sidecar.tmp then renamed. Outer manifest is written to manifest.tmp then renamed LAST: this is the logical commit marker. Failure leaves an uncommitted directory for manual quarantine; automatic cleanup/reuse is intentionally absent. Missing manifest, malformed/canonicalization-invalid manifest, missing/nonregular artifacts, incorrect lengths/hashes, identity/seed/epoch/step/shape discrepancies, invalid sidecar or metadata disagreement fail before restore.

Manifest binds SHA-256 of exact full SCP bytes (including existing checksum), exact sidecar bytes, permutation encoding/algorithm and digest, seed/epoch/step/document/slot counts and identity hashes. Sidecar checkpoint_id is the measured full SCP SHA-256. Permutation digest encodes a domain tag, 64-bit little-endian count and rank-to-canonical entries. The seed-derived permutation is independently regenerated for the current epoch using the existing algorithm, including identity ordering for unshuffled epochs. Header includes a self-contained SHA-256 implementation with empty/abc/multiblock known-answer checks on each entry; parent should independently audit/test it.

Default caps: SCP 2 GiB, sidecar 64 MiB, manifest 4096 bytes. Adjust limits explicitly for legitimate larger models. Full-file reads and snapshots impose substantial host memory overhead; optimize only after correctness review. Save-side SCP cap is checked after the existing serializer writes it, so the limit is not a save disk-quota guarantee.

## Boundaries and caveats

- This is NOT a production integration and has not been compiled or run here. Parent should compile in its native build and exercise new temporary artifacts only. Actual GPU resume, interrupted-training equivalence, and crash-injection tests remain future work.
- Atomic rename provides process-visible logical publication on supported same-filesystem paths; standard C++ streams do not fsync file/directory metadata. No power-loss durability guarantee is made.
- A trusted expected manifest SHA-256 is essential. Existing SCP FNV checksum and plain identity strings are not authentication. No signature/MAC/key management is implemented.
- Parent directories/artifacts must be private and immutable during load. The existing loader reopens SCP by pathname. The wrapper rechecks bytes after parsing, but cannot prevent a hostile transient swap/ABA race or symlink ancestor replacement. Portable standard-library existence checks plus rename are not an OS no-replace guarantee against a hostile concurrent actor. Unique directory reservation protects cooperating writers only.
- State/tensors remain by slot; there is no SharedPtr or tensor permutation. Checkpoint capture uses existing safe-boundary checks and copies cursor metadata only for the mutable-signature workaround.
- All transaction/sidecar/rank checks precede restore. Existing restore can partially mutate GPU state if a device error occurs; this wrapper does not provide GPU rollback atomicity. Do not interpret successful metadata checks as end-to-end resumability proof.
