# Controlled retrieval experiment protocol (before execution)

Purpose: distinguish learnable retrieval from untrained sensitivity and test distance-search bias. This is a separate synthetic task, not a language-model-quality benchmark or replacement production checkpoint.

Use original vocabulary IDs, Q1, reader, Wbi frozen. Generate64-token examples with explicit query token encoding a requested distance d in{2,4,8,15,16,32}; place answer at index63-d. Balance answers and fillers; train/test use disjoint RNG seeds/contexts and an additional disjoint-answer-ID split. Query remains constant for A/B paired variants; other filler unchanged. Record datasets and SHA, explicit metadata rules, no train/test overlap. d64 is an outside-window negative control, cannot provide recoverable answer. No fake EOS from synthetic boundaries and no overwrite of original corpus.

StageA: native CUDA train a small independent readout on frozen actual reader/hash features to test whether features support requested retrieval. Compare frozen production head, new readout, no-key/paired-shuffle controls. Track train/test CE and top1, answer-subset accuracy with balanced labels, multiple seeds. Unseen-answer split must not be used to select parameters; failure to output never-trained classes is expected for an untied classifier and must not be mislabeled as failed memory. Prefer tied vocabulary-row output if claiming unseen-key generalization; otherwise report seen-answer/disjoint-context primary and unseen-answer exploratory separately.

StageB: same fixed trained readout, candidate state gates with joint originalTCG+new distance union<=2, unchanged amplitude budget. Compare ascending order versus seeded randomized order on same train examples/objective/candidate budget, record order and accepted coordinates. Exact-pair exhaustive subset if computationally feasible, labeled differing search budget. Test metrics only after training decisions. Include per-distance usable support, no validation selection.

Acceptance: CPU/GPU state and logits parity on same examples, no artifacts overwriting production, dataset-overlap checks, balanced-label controls. Decisions based on actual evidence; do not expand horizon before meaningful retrieval gain is established.

Current status: protocol only; training and search results NOT yet produced.
