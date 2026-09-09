# Prefill head elision validated

Fixed preserved step360 model. Fresh/continuation/reset checks matched every-token recurrent states and final assistant-role logits bitwise. Prompt15BPEtokens,19totalpositions. Four alternating trials baseline52.24-56.60ms vs advance39.53-42.63ms: about21-30percent lower prefill latency. Excludes loading/tokenization/stateallocation/generation; includes final assistant-role head. Not generated-token speed improvement.

Independent resident executable build/cpu_resident_advance_byte.exe compiled and prompt smoke completed. Original executables preserved. GPU training untouched. See build/advance_prefill_result.log and build/advance_resident_smoke.log for measurements.
