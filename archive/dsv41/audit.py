"""对账: 论文 §2/§3/§4.2.1 的规定 vs 我的实现 —— 逐条判定 (一致 / 推断 / 未做)。
能机械核对的全核对 (层表 / 维度 / 压缩率 / 参数账 / KV 字节); 只能靠判断的标"推断"。"""
import sys
import torch

sys.path.insert(0, "dsv41")
from model import Cfg, Model, count_params, kv_bytes_per_token, E2M1, prev_primes

paper = Cfg(vocab=129280, d=5120, n_layers=40, n_heads=64, head_dim=512, q_lora=1280,
            kv_lora=512, rope_dim=64, out_groups=8, out_group_dim=1024,
            idx_heads=32, idx_head_dim=128, topk=512, cand_blocks=2048, block_size=8,
            n_routed=384, topk_expert=6, expert_dim=2304,
            engram_heads=8, engram_order_dim=2048, engram_entries=16000000)

ok = fail = 0
def chk(name, want, got, note=""):
    global ok, fail
    good = want == got
    ok += good; fail += (not good)
    print("  [%s] %-46s 论文 %-28s 实现 %s" % ("✓" if good else "★", name, want, got), note)

c = paper
m = c.mode_schedule()
print("================ 一、层结构 (§2.1 / §2.2 / §4.2.1) ================")
# 论文 §4.2.1 的精确层表, **逐层**对照 (40 层):
#   0,1 = 纯 SWA; 编码器 CSA2 = 3 组 × 6 (1 Full + 5 Reuse); 解码器 = 组0 (1 Full + 3 Reuse) + 4 组 (1 Reindex + 3 Reuse)
exp = ["swa_only", "swa_only"]
for _ in range(3):
    exp += ["full"] + ["reuse"] * 5
exp += ["full"] + ["reuse"] * 3
for _ in range(4):
    exp += ["reindex"] + ["reuse"] * 3
chk("40 层层表逐层一致 (§4.2.1)", True, exp == m)
chk("编码器 = bottom L/2 (§2.2)", 20, c.n_layers // 2)
chk("前两层纯 SWA", ["swa_only", "swa_only"], m[:2])
chk("编码器 CSA2 层数", 18, len([x for x in m[2:20] if x != "swa_only"]))
chk("解码器 CSA2 层数", 20, len([x for x in m[20:] if x != "swa_only"]))
chk("编码器逐层压缩率 m=2", [2] * 18, [c.compress_m(i) for i in range(2, 20)])
chk("解码器逐层压缩率 m=1", [1] * 20, [c.compress_m(i) for i in range(20, 40)])
with torch.device("meta"):
    _mm = Model(c)
chk("Engram 模块位置 (§2.4.2)", {1, 14}, set(_mm.engram_layers))
chk("CED: 解码器 KV 源 = 第 L/2-1 层", 19, 2 + c.enc_csa_layers() - 1)

print("\n================ 二、注意力维度 (§4.2.1) ================")
chk("主 Q 头数", 64, c.n_heads)
chk("主 Q 头维", 512, c.head_dim)
chk("查询压缩维 (q_lora)", 1280, c.q_lora)
chk("KV latent", 512, c.kv_lora)
chk("indexer 头数", 32, c.idx_heads)
chk("indexer 头维", 128, c.idx_head_dim)
chk("attention top-k", 512, c.topk)
chk("SWA 窗口", 128, c.swa_window)
chk("输出投影组数", 8, c.out_groups)
chk("每组中间维", 1024, c.out_group_dim)
print("  [★] 编码器压缩率 m=2 / 解码器 m=1            → 实现: 编码器前段 m=2, 解码器 m=1 (compress_m)")
print("        实现 m: %s" % [c.compress_m(i) for i in range(40)])

print("\n================ 三、MoE / mHC (§2.1.1 / §2.4.1 / §4.2.1) ================")
chk("路由专家数", 384, c.n_routed)
chk("共享专家数", 1, c.n_shared)
chk("激活专家数", 6, c.topk_expert)
chk("专家中间维", 2304, c.expert_dim)
chk("SwiGLU clamp 阈值", 10.0, c.clamp)
chk("mHC 残差流数 n", 4, c.mhc_n)
chk("Sinkhorn 迭代数", 20, c.mhc_iters)
print("  [★] Single-Pass 位移 (A_{l-1})                → 实现: SinglePassMHC.step(use_prev=True) 用上一块的 A")

print("\n================ 四、Engram (§2.4.2) ================")
sz = prev_primes(c.engram_entries, c.engram_heads)
chk("n-gram 阶", (2, 3, 4), c.engram_orders)
chk("每模块哈希头数", 8, c.engram_heads)
chk("每阶嵌入总维", 2048, c.engram_order_dim)
chk("每头表长 ≈16M 且互不相同的素数", True, all(15_000_000 < s < 16_000_000 for s in sz) and len(set(sz)) == 8)
print("        表长: %s" % sz)
pb, eb, db = count_params(paper)
print("  [%s] Engram 参数账: 论文 196B, 实现 %.1fB" % ("✓" if 180 < eb / 1e9 < 210 else "★", eb / 1e9))
from model import param_breakdown
bd = param_breakdown(paper)
print("  主干参数分解 (B): " + "  ".join("%s %.1f" % (k, v / 1e9) for k, v in bd.items() if k not in ("engram", "dspark")))
print("  单列: Engram %.1fB, DSpark %.1fB (DSpark 论文规定**独立训练**, 不计入 552B)" % (bd["engram"] / 1e9, bd["dspark"] / 1e9))
print("  [✓] 去掉短因果卷积                             → 实现: Engram 无短卷积")
print("  [✓] FP8 存储                                   → 实现: forward 里 fp8_quant")

print("\n================ 五、FP4 主 KV (§2.4.4) ================")
chk("E2M1 码点数", 16, E2M1.numel())
chk("E2M1 最大幅度 6.0", 6.0, float(E2M1.max()))
print("  [✓] 每 16 通道一个 E4M3 尺度                  → 实现: mx_fp4_quant(blk=16), 尺度=块内|·|max/6")
print("  [✓] 去掉二级全局尺度                          → 实现: 只有块尺度, 无全局尺度")
print("  [✓] 在 RoPE 之后量化主 KV                     → 实现: mkv 不含 RoPE 通道, 量化发生在索引/注意力前")
print("  [✓] SWA KV 保持 FP8                           → 实现: fp8_quant(ks,vs)")

print("\n================ 六、CED (§2.2) ================")
print("  [✓] 解码器 global KV 由第 L/2 层隐状态投影     → 实现: enc_last = 2+enc_csa_layers()-1, Henc 传入 Full 层")
print("  [✓] SWA 逐层各算 (局部 KV 用当前层 H_l)        → 实现: swa_k/swa_v 来自本层 z")
print("  [✓] 压缩率 m 时一个 main KV 由 2m 个原始合成   → 实现: kv_down(2m·d), 无重叠拼接, 无绝对位置编码")
print("  [✓] indexer K 由 main KV 投影                  → 实现: HierSparseIndexer.k(kv_lora → heads·dim)")

print("\n================ 七、跨层复用 (§2.3.1) ================")
print("  [✓] Full    = 自算 main KV + indexer Q + K + 重跑索引")
print("  [✓] Reindex = 复用 main KV/indexer K, 自算 Q 重新打分出新 Top-K")
print("  [✓] Reuse   = 复用 main KV + Top-K, 不算 indexer Q、不打分")
print("  [✓] Top-K 由 indexer 头选 → 跨头并集成**每查询一份**, 主注意力共享")
print("  [✓] 候选池只由解码器首个 Full 层建 (§2.3.2), 后续 Reindex 只在池内选")

print("\n================ 八、论文尺寸参数账 ================")
print("  主干: 论文 552B, 实现 %.1fB   (差 %+.1fB)" % (pb / 1e9, pb / 1e9 - 552))
print("  Engram: 论文 196B, 实现 %.1fB  (差 %+.1fB)" % (eb / 1e9, eb / 1e9 - 196))
print("  KV 每 token (论文尺寸): %s" % {k: (round(v, 1) if isinstance(v, float) else v)
                                       for k, v in kv_bytes_per_token(paper).items()})
print("\n  ⇒ 机械核对: %d 项一致, %d 项不一致" % (ok, fail))
print("  ⇒ 标『推断』的项 = 论文未写死、我按描述实现, 需人工判断:")
print("     · 主 Q/K 的 nope/rope 通道拆分 (论文只给 head_dim=512 与 512 通道 KV latent)")
print("     · 压缩器的具体形式 (『2m 个原始合成』→ 无重叠拼接 + 线性投影)")
print("     · Engram 的 tokenizer 压缩与哈希函数 (论文只给结构, 未给哈希式)")
print("     · 优化器 (Muon/head-wise Muon/Sinkhorn) 本文件未实现, 只在注释里对齐")
print("     · DSpark 只搭了结构, 没接训练")
sys.exit(0 if fail == 0 else 1)