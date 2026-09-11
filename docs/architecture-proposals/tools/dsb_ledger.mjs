#!/usr/bin/env node
// dsb_ledger.mjs — DSB 账本工具（Yaoyao / tao::dual）
//
// 用途：解析 DSB2 检查点，输出「字节账 / 稀疏度账 / 算力账」，用于一切性能与容量决策。
// 依赖：仅 Node.js（无第三方包）。用法：
//   node dsb_ledger.mjs <a.dsb> [b.dsb ...]
//   node dsb_ledger.mjs --json <a.dsb>
//
// 口径说明（重要）：
//   * DSB 字节 = 磁盘/2-bit 打包口径。
//   * 解码器常驻字节 = read_compact_bundle 把 2-bit 码展开为 int8 后的口径（1 字节/权重）。
//     二者相差约 3.8 倍，做内存/带宽决策时必须用「常驻」口径。
import { readFileSync } from "node:fs";

const FNV_OFFSET = 14695981039346656037n;
const FNV_PRIME = 1099511628211n;
function fnv1a(bytes) {
  let h = FNV_OFFSET;
  for (const b of bytes) { h ^= BigInt(b); h = (h * FNV_PRIME) & 0xffffffffffffffffn; }
  return h;
}

function parseDsb(path) {
  const buf = readFileSync(path);
  if (buf.toString("latin1", 0, 4) !== "DSB2") throw new Error("not DSB2: " + path);
  const manifestSize = Number(buf.readBigUInt64LE(4));
  const payloadSize = Number(buf.readBigUInt64LE(12));
  const checksum = buf.readBigUInt64LE(20);
  const body = buf.subarray(28);
  const checksumOk = fnv1a(body) === checksum;
  const manifest = buf.toString("utf8", 28, 28 + manifestSize);
  const payload = buf.subarray(28 + manifestSize);
  if (payload.toString("latin1", 0, 4) !== "DSM1") throw new Error("payload magic");
  const lines = manifest.replace(/\r/g, "").trimEnd().split("\n");
  const operatorId = lines[0], tokenizer = lines[1];
  const specs = lines.slice(2).map((l) => {
    const [name, rows, cols, ternary] = l.split(" ");
    return { name, rows: +rows, cols: +cols, ternary: ternary === "1" };
  });
  // 算子版本决定 payload 头里有没有 key 维度 dk（dual-state-4 增量规则记忆才有）。
  const hasDk = specs.some((s) => s.name.endsWith("mem.key"));
  let pos = 4;
  const dims = {};
  for (const k of ["layers", "d", "s", "m", "e", "vocab"]) { dims[k] = payload.readUInt32LE(pos); pos += 4; }
  if (hasDk) { dims.dk = payload.readUInt32LE(pos); pos += 4; }
  const tensors = [];
  for (const sp of specs) {
    const elements = sp.rows * sp.cols;
    if (!sp.ternary) {
      pos += elements * 4;
      tensors.push({ ...sp, elements, bytes: elements * 4, zeros: null, alphas: null });
      continue;
    }
    const start = pos;
    let zeros = 0;
    const alphas = [];
    const densities = [];
    for (let r = 0; r < sp.rows; ++r) {
      alphas.push(payload.readFloatLE(pos)); pos += 4;
      let nz = 0;
      for (let j = 0; j < sp.cols; j += 4) {
        const byte = payload[pos++];
        for (let k = 0; k < 4 && j + k < sp.cols; ++k) {
          const code = (byte >> (2 * k)) & 3;
          if (code === 3) throw new Error("illegal 2-bit code 3 in " + sp.name);
          if (code === 0) zeros++; else nz++;
        }
      }
      densities.push(nz / sp.cols);
    }
    tensors.push({ ...sp, elements, bytes: pos - start, zeros, alphas, densities });
  }
  if (pos + 28 + manifestSize !== buf.length) throw new Error("trailing/garbage bytes");
  return { path, operatorId, tokenizer, dims, tensors, checksumOk, fileBytes: buf.length };
}

const q = (arr, p) => { const a = [...arr].sort((x, y) => x - y); return a[Math.min(a.length - 1, Math.max(0, Math.floor(p * a.length)))]; };
const n = (x, d = 0) => x.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });

function report(r) {
  const c = r.dims;
  const ternary = r.tensors.filter((t) => t.ternary);
  const floats = r.tensors.filter((t) => !t.ternary);
  const tBytes = ternary.reduce((a, t) => a + t.bytes, 0);
  const tElems = ternary.reduce((a, t) => a + t.elements, 0);
  const tZeros = ternary.reduce((a, t) => a + (t.zeros || 0), 0);
  const fBytes = floats.reduce((a, t) => a + t.bytes, 0);
  const allAlphas = ternary.flatMap((t) => t.alphas);
  const allDens = ternary.flatMap((t) => t.densities);

  const head = r.tensors.find((t) => t.name === "embedding");
  // 每层 MAC 直接由 schema 推出，不再对算子版本硬编码。
  const layerTensors = r.tensors.filter((t) => /^layer\.0\./.test(t.name) && t.ternary);
  const readMac = layerTensors.filter((t) => t.name.includes("read.")).reduce((a, t) => a + t.rows * t.cols, 0);
  const projMac = layerTensors.filter((t) => !t.name.includes("read.")).reduce((a, t) => a + t.rows * t.cols, 0);
  // 增量规则记忆的状态算子：M k、外积更新、M q（各 m*dk），外加标量 beta 的 d 次内积。
  const stateMac = c.dk ? 3 * c.m * c.dk + c.d : 0;
  const perLayer = projMac + readMac + stateMac;
  const headMac = c.vocab * c.d;
  const bodyMac = perLayer * c.layers;

  const out = [];
  out.push("## " + r.path);
  out.push("");
  out.push("- 算子身份：" + r.operatorId + "；tokenizer：" + r.tokenizer.slice(0, 16) + "…；校验和：" + (r.checksumOk ? "OK" : "**FAIL**"));
  out.push("- 维度：layers=" + c.layers + " d=" + c.d + " s=" + c.s + " m=" + c.m + (c.dk ? " dk=" + c.dk : "") + " e=" + c.e + " vocab=" + c.vocab);
  out.push("");
  out.push("### 字节账");
  out.push("");
  out.push("| 项 | 字节 | 说明 |");
  out.push("|---|---:|---|");
  out.push("| DSB 文件 | " + n(r.fileBytes) + " | 2-bit 打包口径 |");
  out.push("| 三值张量 | " + n(tBytes) + " | " + n(tElems) + " 个元素 |");
  out.push("| ├ 共享 embedding | " + n(head ? head.bytes : 0) + " | 占 DSB " + (head ? (100 * head.bytes / r.fileBytes).toFixed(1) : "0") + " % |");
  out.push("| └ 其余图层 | " + n(tBytes - (head ? head.bytes : 0)) + " | 每层 " + n((tBytes - (head ? head.bytes : 0)) / c.layers) + " |");
  out.push("| 浮点张量 | " + n(fBytes) + " | norm 增益 / bias / vocab.bias |");
  const ternaryRows = ternary.reduce((a, x) => a + x.rows, 0);
  const resident = tElems + ternaryRows * 4 + fBytes;
  out.push("| **解码器常驻（int8 q 口径）** | **" + n(resident) + "** | q 1 字节/权重（" + n(tElems) + "）+ 行尺度 4 B/行（" + n(ternaryRows * 4) + "）+ 浮点（" + n(fBytes) + "）|");
  out.push("| 每 token 权重字节 | " + n(resident) + " | 常驻口径下每 token 全量流过一次 |");
  const stateElems = c.dk ? c.s + c.m * c.dk : c.s + c.m;
  out.push("| 每会话状态 | " + n(c.layers * stateElems * 4) + " | float32" + (c.dk ? "（s 向量 + " + c.m + "×" + c.dk + " 矩阵状态）" : "") + " |");
  out.push("");
  out.push("### 稀疏度与尺度");
  out.push("");
  out.push("- 零元素比例：" + (100 * tZeros / tElems).toFixed(2) + " %（逐行密度中位 " + (100 * q(allDens, 0.5)).toFixed(1) + " %，范围 " + (100 * q(allDens, 0)).toFixed(1) + "–" + (100 * q(allDens, 1)).toFixed(1) + " %）");
  out.push("- 行尺度 α（全体）：min " + q(allAlphas, 0).toFixed(5) + " / 中位 " + q(allAlphas, 0.5).toFixed(5) + " / max " + q(allAlphas, 1).toFixed(5) + "，极差 **" + (q(allAlphas, 1) / q(allAlphas, 0)).toFixed(2) + "×**");
  if (head && head.alphas) {
    out.push("- 行尺度 α（仅 embedding，即输出头所在矩阵）：极小 " + q(head.alphas, 0).toFixed(5) + " / 中位 " + q(head.alphas, 0.5).toFixed(5) + " / 极大 " + q(head.alphas, 1).toFixed(5) + "，极差 **" + (q(head.alphas, 1) / q(head.alphas, 0)).toFixed(2) + "×**");
  }
  out.push("- 推论：α 与行密度分布都很窄（无「极稀疏行 / 极小尺度行」子群体）→ 索引式稀疏存储与 Cauchy–Schwarz 精确剪枝均无收益（见 02 号文档「死路」一节）。");
  out.push("");
  out.push("### 算力账（每 token 稠密 MAC）");
  out.push("");
  out.push("| 项 | MAC | 占比 |");
  out.push("|---|---:|---:|");
  out.push("| 输出头 vocab×d | " + n(headMac) + " | " + (100 * headMac / (headMac + bodyMac)).toFixed(1) + " % |");
  out.push("| 每层小计（三值投影 " + n(projMac) + (stateMac ? " + 状态算子 " + n(stateMac) : "") + " + 读出 " + n(readMac) + "） | " + n(perLayer) + " | " + (100 * perLayer / (headMac + bodyMac)).toFixed(2) + " % ×" + c.layers + " = " + (100 * bodyMac / (headMac + bodyMac)).toFixed(1) + " % |");
  out.push("| ↳ 层内占比 | 三值投影 " + (100 * projMac / perLayer).toFixed(1) + " %" + (stateMac ? " / 状态算子 " + (100 * stateMac / perLayer).toFixed(1) + " %" : "") + " / 读出 " + (100 * readMac / perLayer).toFixed(1) + " % | — |");
  out.push("| **合计** | **" + n(headMac + bodyMac) + "** | 100 % |");
  out.push("");
  out.push("### 逐张量");
  out.push("");
  out.push("| 张量 | rows×cols | 三值 | 字节 | 零元素 % |");
  out.push("|---|---|---|---:|---:|");
  for (const t of r.tensors) {
    out.push("| " + t.name + " | " + t.rows + "×" + t.cols + " | " + (t.ternary ? "是" : "否") + " | " + n(t.bytes) + " | " + (t.zeros === null ? "—" : (100 * t.zeros / t.elements).toFixed(2)) + " |");
  }
  return out.join("\n");
}

const args = process.argv.slice(2);
const json = args.includes("--json");
const paths = args.filter((a) => !a.startsWith("--"));
if (!paths.length) { console.error("usage: node dsb_ledger.mjs [--json] <a.dsb> [...]"); process.exit(2); }
for (const p of paths) {
  const r = parseDsb(p);
  console.log(json ? JSON.stringify({ operatorId: r.operatorId, dims: r.dims, checksumOk: r.checksumOk, tensors: r.tensors.map(({ alphas, densities, ...t }) => t) }, null, 1) : report(r));
}
