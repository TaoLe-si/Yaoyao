# Filtered long-run status

Process 34952 started with lr=.0001, Adam reset, filtered 19073-doc corpus, official 23-doc GPU val every update, budget 200. Production STOP retained. Unique dir build/filtered_repair_v1, no overwrite of 1200/1204 or expanded_repair_v1.

Initial val NLL 6.309478603453 (matches completed 1240 master). Early updates finite and declining at last observed step 3 (6.25584). Not a completed experiment; do not claim oscillation cured.

After FINAL step=200, run once: D:/nodejs/node.exe test_filtered_repair_quality.mjs. Same 9 prompts, refuse overwrite. Lower NLL without readable answers is not success.
