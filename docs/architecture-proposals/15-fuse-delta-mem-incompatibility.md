# 15 · 融合（fuse_sm）与 H2R 增量记忆不兼容：一个潜伏了三轮的缺陷

> 本轮把 `dual-state-4`（H2R 增量规则矩阵记忆）拿到 L4 语料上重跑时，
> **默认配置下 CPU 解码器直接崩溃**（`FAIL invalid map<K, T> key`）。
> 定位后发现这是一个**组合型潜伏缺陷**：两个各自正确的特性放在一起会互相破坏。
> 本文记录根因、修复、以及它对既有结论的影响。

---

## 一、症状

```
> build\h2r_cpu.exe build\h2r_l4_run\step_0\final.dsb
RESET
READY loads=1 load_seconds=0.052
FAIL invalid map<K, T> key
```

- **不是**「模型坏了」：同一个 exe 读**旧的** `build\h2r_consistency.dsb`、`build\delta_mem_smoke.dsb` 也失败
  → 说明是**代码路径**问题，不是某份产物的问题
- **不是** bundle 格式问题：`read_compact_bundle` 的 operator identity / schema / checksum 校验**全部通过**
  （否则会抛 `operator identity mismatch` 或 `schema/identity`），构造也完成了

## 二、根因

`GreedyPipelineGroupedModel::recurrent()` 的**层内 dispatch 融合**（doc 02 §44.6）：

```cpp
const bool can_fuse=fuse_sm_&&!(reuse_mv&&!cachedV.empty()&&layerSource[l]!=l);
...
if(can_fuse){ ... group<10>({p+"s.candidate.x", p+"s.candidate.s", p+"s.gate.x", p+"s.gate.s",
                              p+"m.candidate.x", p+"m.candidate.s", p+"m.candidate.m",
                              p+"m.gate.x",      p+"m.gate.s",      p+"m.gate.m"}, ...); }
```

融合把 **s 分支（4 个矩阵）与 m 分支（6 个矩阵）合并成一次 dispatch**，为此显式列出了
`m.candidate.*` / `m.gate.*` 共 6 个张量名。

**但 `TAO_DELTA_MEM` 的 schema 里根本没有这些张量。** 对照 `dual_state_config.hpp`：

```cpp
#ifdef TAO_DELTA_MEM
add(p+"mem.key",c.dk,c.d); add(p+"mem.query",c.dk,c.d); add(p+"mem.value",c.m,c.d);
add(p+"mem.beta",1,c.d,false); add(p+"mem.beta.bias",1,1,false);
#else
for(auto branch:{"m.candidate","m.gate"}){ add(p+branch+".x",c.m,c.d); ... }   // ← 融合依赖的这 6 个
#endif
```

于是 `group<10>` 对不存在的键调用 `packed.at()` → `std::out_of_range` → MSVC 的消息正是
**"invalid map<K, T> key"**。

**决定性验证**（同一份产物、同一个 exe，只改一个环境变量）：

| 配置 | 结果 |
|---|---|
| `TAO_FUSE_SM=0`（关闭融合） | **status=0，正常解码** |
| 默认（融合开启） | **status=1，`invalid map<K, T> key`** |

## 三、为什么潜伏了三轮没被发现

1. **融合是后加的**（doc 02 §44.6），而 H2R 臂（`dual-state-4`）的实验更早（doc 12）。
   两者**从未一起跑过**——加融合时只回归了 `dual-state-3`。
2. **`fuse_sm_` 默认开启**（成员初值 `true`，且构造函数 `envOn("TAO_FUSE_SM",true)`），
   所以「忘了测」的代价是**默认路径直接崩**，而不是性能回退。
3. **生产解码器 `yaoyao_cpu_v01.exe` 不定义 `TAO_DELTA_MEM`**，因此生产路径完全不受影响；
   缺陷只出现在 `h2r_*` 这一组诊断/实验二进制里，长期没人跑。
4. `h2r_diag.exe` 崩溃时返回 `0xC0000409`（STATUS_STACK_BUFFER_OVERRUN）且**零输出**，
   比 `h2r_cpu.exe` 的异常消息更难诊断。

## 四、修复

`src/greedy_pipeline_grouped_model.hpp`：

```cpp
// 【TAO_DELTA_MEM 下禁止融合】融合组引用 m.candidate.*/m.gate.*，而 dual-state-4 的
// schema 用 mem.key/query/value 取代了它们。若照常融合，group<10> 会对不存在的键
// 调用 packed.at()，抛 "invalid map<K, T> key"（实测：默认崩溃，TAO_FUSE_SM=0 正常）。
#ifdef TAO_DELTA_MEM
const bool can_fuse=false;
#else
const bool can_fuse=fuse_sm_&&!(reuse_mv&&!cachedV.empty()&&layerSource[l]!=l);
#endif
```

修复后：`build\h2r_cpu.exe build\h2r_consistency.dsb` → **status=0**（默认环境，融合开启）。

> 说明：`dual-state-4` 下 m 分支本身已被 `mem.*` 取代，s/m 融合在语义上也不适用，
> 因此「禁用」而不是「改写融合组」是正确的修法——不会损失任何有效优化。

## 五、对既有结论的影响

| 文档 | 影响 |
|---|---|
| doc 12（H2R 臂判定） | **不受影响**：那轮 H2R 是能跑的（否则不会有数字）。但必须承认：**若当时融合已开启，这组数字根本产生不出来**。本轮已修复，H2R 可以正常重跑。 |
| doc 14（绑定/复制判定） | **不受影响**：doc 14 的结论基于 `dual-state-3`（无 `TAO_DELTA_MEM`），与该缺陷无关。 |
| doc 02 §44.6（融合） | **结论需加限定**：融合的收益只在 `dual-state-3` 上测过；`dual-state-4` 下不可用。 |

## 六、纪律教训（第四次同类）

> **「两个特性各自正确」不代表「组合正确」。特性组合必须显式加入回归矩阵。**

前三轮同类教训都是「开关没生效就下结论」（死环境变量、head 扫描、层数扫描）。
这次方向相反：**开关生效了，但生效在一个不该生效的地方**。
可操作的防护：凡是 `#ifdef` 分叉出的 schema，**所有依赖具体张量名的代码路径都必须带同样的 `#ifdef`**。
`group<10>` 硬编码了 10 个张量名，是这类脆弱点的典型。

## 七、第二个融合缺陷：默认值与训练语义不一致（违反 I4）

同一次排查中还发现 `fuse_sm_` 的**默认值**本身就是错的。

融合的语义是 **mHC 式依赖错位**——m 分支改读本层**更新前**的 `s`（`s_old`）：

```cpp
if(can_fuse){s_old=s;auto st10=group<10>({..., p+"m.candidate.s", ...}, {..., &s_old, ...});}
```

但训练器 `dual_state_autograd.cuh` **没有融合路径**：

```cpp
// 136 行：先算 s
auto u=branch("s.candidate",false),a=branch("s.gate",false);s[l]=tape.update(s[l],u,a);
// 146 行：m 分支读的是**更新后**的 s[l]
auto v=branch("m.candidate",true),g=branch("m.gate",true);m[l]=tape.update(m[l],v,g);
```

`grep fuse` 在训练器里只命中一个无关的 `TAO_FUSED_ADD_BACKWARD`。
**所以「训练期同域」的旧注释不成立：训练是顺序依赖，推理默认是错位依赖，不是同一算子。**

**修复**：默认改为关闭。

```cpp
applyDefaultArch();fuse_sm_=envOn("TAO_FUSE_SM",false);
```

理由三条：

1. **违反 I4**（本项目明文不变量：训练/推理同算子）
2. **收益已被证伪**：§44.6 实测 **0.991×**（无加速），关掉不损失性能
3. **关闭后实测更好**（L4 step400，12 个 held-out 名字）：

| 配置 | 名字 | 颜色 | 首两条输出 |
|---|---:|---:|---|
| 新默认（fuse off） | **2/12** | **5/12** | `邹晓东，紫色。` `鲍刚。` |
| 显式 `TAO_FUSE_SM=1` | 1/12 | 4/12 | `��军，，晨。` `鲍刚。` |
| 显式 `TAO_FUSE_SM=0` | 2/12 | 5/12 | 同新默认 ✓ |

注意 `邹晓东，紫色。`：held-out 名字是 `邬晓东`，模型把**名字部分（晓东）与颜色都抄对了**。

---

## 八、复现

```bat
cd /d D:\TaoVm
:: 复现崩溃（默认融合开启）
build\h2r_cpu.exe build\h2r_consistency.dsb
:: 确认是融合引起
set TAO_FUSE_SM=0
build\h2r_cpu.exe build\h2r_consistency.dsb
:: 查看实际留下的张量（dual-state-4 只有 layer.0/layer.4 的 mem.* 与 s.*）
set TAO_WEIGHT_DUMP=1
build\h2r_cpu.exe build\h2r_consistency.dsb
:: 验证 I4 修复后的默认值
build\yaoyao_cpu_v01.exe build\noffn_l4_run\step_400\final.dsb
```
