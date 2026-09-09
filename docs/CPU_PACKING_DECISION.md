# CPU packing selection

Measured on preserved step360 full dual-state model while GPU training runs. Two-bit shifts and LUT both passed828synthetic row tests across92widths and32step full logits/state/argmax exactness.

Shift candidate fullgreedy238-263positions/s vs byte325-374. LUT candidate298-307 vs byte338-349. LUT improves unpack overhead but remains9-14percent slower than byte within alternating trials. Do not deploy either as speed improvement; their7,956,480byte payload plus shared4096byte LUT (LUT variant) is a memory tradeoff only.

Current preferred path: byte ternary AVX2 with direct compact loading. Next investigation: prebound tensor references and reusable scratch buffers. No CPU training or GPU interference commands.
