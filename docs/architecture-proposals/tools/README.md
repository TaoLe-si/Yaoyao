# tools · 架构决策用的小工具

| 工具 | 用途 | 依赖 | 用法 |
|---|---|---|---|
| `dsb_ledger.mjs` | 解析 DSB2 检查点，输出**字节账 / 稀疏度与尺度 / 算力账 / 逐张量表**，并校验 FNV-1a 校验和 | Node.js（无第三方包） | `node dsb_ledger.mjs <a.dsb> [b.dsb ...]`；加 `--json` 输出机器可读 |
| `cpuid_probe.cpp` | 打印 AVX-512 子特性与 L2/L3 大小（决定能否用 `vpermb`/`vpdpbusd`） | MSVC | 见文件头注释 |
| `extract_pdf.py` | 论文 PDF → 带页分隔符的文本（核对原文数字用） | pypdf | `python extract_pdf.py <in.pdf> <out.txt>` |

## 口径提醒

`dsb_ledger.mjs` 同时给出两个口径，**不要混用**：

- **DSB 打包口径**（磁盘/传输）：2 bit/权重，本机基线 5,798,167 B；
- **解码器常驻口径**（内存/带宽）：int8 展开 1 B/权重，本机基线 22,112,256 B。

做内存或带宽结论时一律用**常驻口径**；早期版本按打包口径断言「整模型可常驻 16 MB L3」，该结论已作废。

## 基线参考值（build/noffn_fresh/step_1512/final.dsb，2026-09-10）

- 算子身份：`dual-state-3-noffn-input-sqrt-d`；校验和 OK；layers=8 d=512 s=128 m=512 e=1024 vocab=16384
- DSB 5,798,167 B；常驻 22,112,256 B；每会话状态 20,480 B
- 零元素 45.82 %；逐行密度中位 54.3 %（27.3–73.4 %）
- 行尺度 α：全体 0.0283–0.1020（3.61×）；仅 embedding 0.0388–0.0836（2.16×）
- 每 token MAC 21,757,952（输出头 38.6 %；每层 7.68 % × 8 = 61.4 %）
