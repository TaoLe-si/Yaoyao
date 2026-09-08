# Expanded supported-token contextual Wbi gates

User authorized expand beyond4 high-frequency tokens before unfreezing head/Wbi. train_tao_context_supported.cu identical to initial contextual trainer except loop over all1024 rows; support threshold64 appearances in four fixed4096-target batches. Starts from zero gates for reproducible matched protocol; first4 rows expected reproduce prior run. Same frozen model47002 and original reader, max2 active distances per token, total absolute coefficient<=1/4, q/32 q in0,+/-1,+/-2,+/-4,+/-8. No validation candidate selection.

Important bounds:64 counts is overall current-token support, not a guarantee64 occurrences with a given distance available. Near64-step boundaries some distances have less support. Further rounds should use per-distance counts before relaxing support or rare-token training. Independent count: among40 eligible rows,11 have fewer than64 usable appearances at distance15; minimum50. These remain in this run under total-count protocol, not falsely classified as per-distance>=64. Each row greedily visits distances1..15, so early distances can occupy two slots; not exact search over all distance pairs. Independent row objectives conditional on current token permit later parallel optimization but current implementation sequential.

Independent binary corpus count:40 eligible token rows cover10635/16384 training positions (64.91%).

Expanded run target tao_context_supported.tcg/log/csv. Context table15KiB; zero unsupported rows. CPU inference fullWbi lookup, no new vocabulary. Report fraction training positions covered, token rows eligible, nonzero gates, three validation CEs. This is same-corpus validation already monitored, not unseen final test or proof of broad language reasoning.
