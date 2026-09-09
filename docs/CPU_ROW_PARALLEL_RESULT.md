# Persistent two-thread decoder result

All executor recovery/kernel tests and32step full logits/states/argmax bitwise comparison passed. Two alternating128-token pairs: single350.897/351.157positions/s, dual424.131/383.992; mean351.027 vs404.062(~15.1percent gain). Preserved step360bundle, explicitAVX2 strict FP, no model changes. CPU/GPUconcurrentload limits inference about stable performance. No deployment yet; worker shuts down before weight storage. Next bounded comparison checks4threads and larger dispatchthreshold rather than unbounded core use.
