# tao_d256_api: OpenAI-compatible CPU streaming server

Serves tao_fixed_step50002.bin + tao_context_supported.tcg + tao_coef_rts1_128.reader. CPU only, no CUDA dependency at inference.

Protocol stdin line: prompt|stream|max_tokens|T|p (yaoyao_api.py, 5 fields) or prompt|max_tokens|T|p (4 fields). Output per token: TOKEN id step_ms total_ms word (spaces encoded 0x01), then DONE. Ready line: Server ready vocab=1024 step=50002.

Startup validation: model magic/version/V, TCG1 header+CRC+gate sparsity/budget, RTS1 magic/version/dims+CRC, corpus SHA256 binding to train_tokens.bin (API passes it).

Intended feature contract (full parity still requires verification): sliding 64-token window with per-window fresh state; mod3 chain prefix via SelfDecodingNode; 17-row ternary reader mixture (inverse_reader.hpp semantics: mix current state, pop until count exhausted, final row mixes state before oldest retained token); hash features = same hash nibble cycle i&7 over 64 features; head = SwiGLU(Wg,Wu) then Wout plus Wbi[prev] and trained contextual Wbi gate branch; production repetition penalty (3.0*0.65^back, last6); EOS/UNK logits suppressed to -1e9 (deviation from old generator, prevents degenerate stop); per-request time-seeded rng.

Bug history fixed this round: (1) service originally summed raw token codes as state and omitted reader coefficients entirely -> collapsed repetitive sampling; (2) hash features used rolling h instead of nibble cycle; (3) request parser treated 5-field API protocol as 4 fields, making max_tokens=1 and T=200 through FastAPI; (4) reader path argument order. All verified by direct stdin run and full FastAPI OpenAI stream.

Earlier OpenAI timing (~0.39 ms/token) used a defective Wbi loader and cannot validate model quality. Audit found Wbi read from optimizer state, different vocabulary frequency-tie ordering/lowercasing, and missing nucleus renormalization. These are corrected in source. test_tao_server_checkpoint.cpp verifies 1048576 Wbi entries, zero current head biases, sampling behavior, and 4096 token IDs against the corpus. Isolated corrected build tao_d256_api_verified.exe generated 64 tokens in 26.660 ms cumulative internal time; see tao_server_baseline_verified.log. This is not HTTP latency. The sample remains incoherent; its cause is not established by these checks. Live replacement and full forward parity subsequently passed with version2 state gates. See TAO_STATE_V2_RESULT.md and openai_state_v2_stream.log for current results.

Files: tao_d256_api.cpp/.exe, yaoyao_api.py (wired, port 11434), openai_client.py, api_openai.log, openai_stream.log, raw_server_run.log.
