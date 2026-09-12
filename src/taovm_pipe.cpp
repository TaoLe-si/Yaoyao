// taovm_pipe.cpp —— Taovm 原生 C++ 训练流水线驱动器
//
// 存在理由：此前的编排由 PowerShell + Python 承担，属于胶水层。本文件把
// 「语料组装 / 去污染 / 拒绝采样 / 阶段编排 / 产物验证」全部改为原生 C++，
// 零脚本依赖。子进程通过 CreateProcess 启动并重定向到各阶段日志。
//
// 用法:
//   taovm_pipe all                 完整流水线（可重复执行，已完成阶段自动跳过）
//   taovm_pipe decontam            仅去污染
//   taovm_pipe assemble            仅组装课程语料
//   taovm_pipe reject IN.tsv OUT   仅拒绝采样
//   taovm_pipe stage NAME          跑单阶段
#include <windows.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <ctime>
#include <iterator>

namespace fs = std::filesystem;
static const std::string ROOT = "D:\\TaoVm";

// ───────────────────────── 日志 ─────────────────────────
static std::string ts() {
    std::time_t t = std::time(nullptr); std::tm lt{}; localtime_s(&lt, &t);
    char b[32]; std::strftime(b, sizeof(b), "%Y-%m-%d %H:%M:%S", &lt); return b;
}
static FILE* g_log = nullptr;
static void say(const std::string& m) {
    std::string line = ts() + "  " + m;
    std::cout << line << std::endl;
    if (g_log) { std::fputs(line.c_str(), g_log); std::fputc('\n', g_log); std::fflush(g_log); }
}

// ───────────────────────── 环境 ─────────────────────────
static void env(const char* k, const std::string& v) { _putenv_s(k, v.c_str()); }
static void unenv(const char* k) { _putenv_s(k, ""); }

// ───────────────────────── 子进程 ─────────────────────────
static std::string q(const std::string& s) { return "\"" + s + "\""; }

static int run_proc(const std::string& cmdline, const fs::path& logpath) {
    fs::create_directories(logpath.parent_path());
    SECURITY_ATTRIBUTES sa{ sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE };
    HANDLE h = CreateFileA(logpath.string().c_str(), FILE_APPEND_DATA,
        FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) { say("ERROR 无法打开日志 " + logpath.string()); return -1; }
    STARTUPINFOA si{}; si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = h; si.hStdError = h; si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    PROCESS_INFORMATION pi{};
    std::vector<char> buf(cmdline.begin(), cmdline.end()); buf.push_back('\0');
    say("EXEC " + cmdline.substr(0, 300));
    BOOL ok = CreateProcessA(nullptr, buf.data(), nullptr, nullptr, TRUE,
                             CREATE_NO_WINDOW, nullptr, ROOT.c_str(), &si, &pi);
    if (!ok) { say("ERROR CreateProcess failed " + std::to_string(GetLastError())); CloseHandle(h); return -1; }
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD ec = 0; GetExitCodeProcess(pi.hProcess, &ec);
    CloseHandle(pi.hThread); CloseHandle(pi.hProcess); CloseHandle(h);
    say("EXIT " + std::to_string(ec));
    return (int)ec;
}

static bool exists(const std::string& p) { std::error_code ec; return fs::exists(fs::path(p), ec); }
static uintmax_t fsize(const std::string& p) { std::error_code ec; auto s = fs::file_size(fs::path(p), ec); return ec ? 0 : s; }

// ───────────────────────── 文本归一化 ─────────────────────────
// 去污染与答案比对共用：剔除空白、半角与全角标点，使「同一道题的不同书写」可比。
static const char* PUNCT[] = {
    "\xEF\xBC\x8C", // ，
    "\xEF\xBC\x8E", // ．
    "\xE3\x80\x82", // 。
    "\xE3\x80\x81", // 、
    "\xEF\xBC\x9B", // ；
    "\xEF\xBC\x9A", // ：
    "\xEF\xBC\x9F", // ？
    "\xEF\xBC\x81", // ！
    "\xEF\xBC\x88", "\xEF\xBC\x89", // （）
};
static std::string norm_text(const std::string& s) {
    std::string o; o.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        unsigned char c = (unsigned char)s[i];
        if (c==' '||c=='\t'||c=='\r'||c=='\n'||c==','||c=='.'||c=='_'||c=='\'') { ++i; continue; }
        bool skip = false;
        if (c == 0xEF || c == 0xE3) {
            for (const char* p : PUNCT) {
                size_t n = std::strlen(p);
                if (s.compare(i, n, p) == 0) { i += n; skip = true; break; }
            }
        }
        if (skip) continue;
        o.push_back(s[i++]);
    }
    return o;
}

// ───────────────────────── 极简 JSON 字段抽取 ─────────────────────────
static std::string json_str(const std::string& line, const std::string& key) {
    std::string pat = "\"" + key + "\"";
    size_t k = line.find(pat);
    if (k == std::string::npos) return "";
    size_t c = line.find(':', k + pat.size());
    if (c == std::string::npos) return "";
    size_t i = c + 1;
    while (i < line.size() && (line[i]==' '||line[i]=='\t')) ++i;
    if (i >= line.size() || line[i] != '"') return "";
    ++i; std::string out;
    while (i < line.size()) {
        char ch = line[i];
        if (ch == '\\') {
            if (i + 1 >= line.size()) break;
            char nx = line[i+1];
            if (nx=='n') out.push_back('\n');
            else if (nx=='t') out.push_back('\t');
            else if (nx=='r') out.push_back('\r');
            else if (nx=='u') { // \uXXXX -> 原样保留转义（中文在语料中通常是裸 UTF-8）
                if (i+5 < line.size()) i += 4;
            } else out.push_back(nx);
            i += 2; continue;
        }
        if (ch == '"') break;
        out.push_back(ch); ++i;
    }
    return out;
}

// ───────────────────────── 载入测试集题干（用于去污染）─────────────────────────
static std::set<std::string> load_test_questions(const std::string& p) {
    std::set<std::string> s;
    std::ifstream f(p, std::ios::binary);
    if (!f) { say("WARN 无法读取测试集 " + p); return s; }
    std::string L;
    while (std::getline(f, L)) {
        if (!L.empty() && L.back()=='\r') L.pop_back();
        if (L.empty()) continue;
        std::string qs = json_str(L, "q");
        if (!qs.empty()) s.insert(norm_text(qs));
    }
    return s;
}

// ─────────────────────── 阶段 1a：去污染 ───────────────────────
// 为什么必须做：预训练语料若含测试题，评测分数会被记忆抬升，实验结论失效。
// 做法：以 grpo_test.jsonl 的题干为准，从推理语料中整篇剔除。
static int cmd_decontam() {
    const std::string src = ROOT + "\\data\\p1_reason.txt";
    const std::string dst = ROOT + "\\data\\p1_reason_clean.txt";
    const std::string tst = ROOT + "\\data\\grpo_test.jsonl";
    auto bad = load_test_questions(tst);
    say("DECONTAM 测试题干数=" + std::to_string(bad.size()));
    std::ifstream f(src, std::ios::binary);
    if (!f) { say("FAIL 无法读取 " + src); return 1; }
    std::ofstream o(dst, std::ios::binary);
    if (!o) { say("FAIL 无法写入 " + dst); return 1; }
    std::string L;
    std::vector<std::string> doc;
    size_t total = 0, dropped = 0;
    auto flush = [&]() {
        if (doc.empty()) return;
        ++total;
        std::string nq;
        for (auto& d : doc) { if (d.size() > 2 && d[0]=='U' && d[1]==' ') { nq = norm_text(d.substr(2)); break; } }
        if (!nq.empty() && bad.count(nq)) { ++dropped; return; }
        for (auto& d : doc) o << d << '\n';
    };
    bool any = false;
    while (std::getline(f, L)) {
        if (!L.empty() && L.back()=='\r') L.pop_back();
        if (L == "DOC") { flush(); doc.clear(); doc.push_back(L); any = true; continue; }
        doc.push_back(L);
    }
    flush();
    o.flush();
    say("DECONTAM 文档总数=" + std::to_string(total) + " 剔除=" + std::to_string(dropped) +
        " 保留=" + std::to_string(total - dropped) + " -> " + dst);
    (void)any;
    return (total - dropped) > 0 ? 0 : 1;
}

// ─────────────────────── 阶段 1b：语料分片 ───────────────────────
struct CorpusJob { std::string name, src, out, stage, sep; bool dlg; };

static int build_shard(const CorpusJob& j) {
    // 跳过判据必须是「上次完整跑完」，不能只看目录非空：
    // 中断留下的部分分片若被当作已完成，剩余文档会被永久跳过。
    {
        std::ifstream lg(ROOT + "\\build\\corpus_" + j.name + ".log", std::ios::binary);
        std::string all((std::istreambuf_iterator<char>(lg)), std::istreambuf_iterator<char>());
        if (all.find("CORPUS_DONE") != std::string::npos && exists(j.out)) {
            int n = 0;
            for (auto& e : fs::directory_iterator(fs::path(j.out), std::error_code{})) if (e.path().extension() == ".bin") ++n;
            if (n > 0) { say("SKIP 分片 " + j.name + " 已完成 " + std::to_string(n) + " 片"); return 0; }
        }
        fs::remove_all(j.out);   // 清理可能的残留，保证重新完整构建
    }
    std::string cmd = q(ROOT + "\\build\\build_corpus.exe") +
        " --out " + q(j.out) + " --tokenizer " + q(ROOT + "\\build\\tok_v2.bbp") +
        " --format " + std::string(j.dlg ? "dialogue" : "plain") +
        " --doc-sep " + q(j.sep) + " --preserve-layout --stage " + j.stage +
        " --min-bytes 32 --docs-per-shard 20000 " + q(j.src);
    return run_proc(cmd, ROOT + "\\build\\corpus_" + j.name + ".log");
}

static std::vector<std::string> list_shards(const std::string& dir) {
    std::vector<std::string> v;
    std::error_code ec;
    for (auto& e : fs::directory_iterator(dir, ec)) {
        if (e.path().extension() == ".bin") v.push_back(e.path().string());
    }
    std::sort(v.begin(), v.end());
    return v;
}

static int cmd_corpus_stage() {
    // 分隔符必须逐文件匹配：wiki/code 用 @@DOC@@，reason/alpaca 用独立行 DOC。
    std::vector<CorpusJob> J = {
        {"wiki",   "E:\\taovm-data\\wiki_docs.txt", ROOT + "\\data\\p1_wiki",   "mixed",     "@@DOC@@", false},
        {"reason", ROOT + "\\data\\p1_reason_clean.txt", ROOT + "\\data\\p1_reason", "core",  "DOC",     true},
        {"code",   "E:\\taovm-data\\code_docs.txt", ROOT + "\\data\\p1_code",   "interfere", "@@DOC@@", false},
        {"alpaca", ROOT + "\\data\\alpaca_zh\\conversations.txt", ROOT + "\\data\\p1_alpaca", "core", "DOC", true},
    };
    for (auto& j : J) { int rc = build_shard(j); if (rc != 0) { say("FAIL 分片失败 " + j.name); return rc; } }
    return 0;
}

// ─────────────────────── 阶段 1c：课程组装 ───────────────────────
static int cmd_assemble() {
    auto W = list_shards(ROOT + "\\data\\p1_wiki");
    auto R = list_shards(ROOT + "\\data\\p1_reason");
    auto C = list_shards(ROOT + "\\data\\p1_code");
    auto A = list_shards(ROOT + "\\data\\p1_alpaca");
    say("ASSEMBLE wiki=" + std::to_string(W.size()) + " reason=" + std::to_string(R.size()) +
        " code=" + std::to_string(C.size()) + " alpaca=" + std::to_string(A.size()));
    if (W.empty() || R.empty()) { say("FAIL 缺少必要分片"); return 1; }

    auto link = [&](const std::vector<std::string>& plan, const std::string& dest, const char* tag) -> int {
        fs::remove_all(dest);
        fs::create_directories(dest);
        int i = 0;
        for (auto& s : plan) {
            char nm[32]; std::snprintf(nm, sizeof(nm), "shard_%05d.bin", i);
            std::string target = s; for (auto& c : target) if (c == '/') c = '\\';
            std::string dstp = dest + "\\" + nm;
            if (!CreateHardLinkA(dstp.c_str(), target.c_str(), nullptr)) {
                std::error_code ec; fs::copy_file(target, dstp, fs::copy_options::overwrite_existing, ec);
                if (ec) { say("FAIL 硬链接/复制失败 " + target); return -1; }
            }
            ++i;
        }
        say(std::string("ASSEMBLE ") + tag + " 分片数=" + std::to_string(i));
        return i;
    };

    // 预训练：推理数据交错分布（堆末尾会让模型把推理当收尾格式）
    size_t wh = W.size() / 2, rh = R.size() / 2;
    std::vector<std::string> p1;
    for (size_t i = 0; i < wh; ++i) p1.push_back(W[i]);
    for (size_t i = 0; i < rh; ++i) p1.push_back(R[i]);
    for (auto& s : C) p1.push_back(s);
    for (size_t i = rh; i < R.size(); ++i) p1.push_back(R[i]);
    for (size_t i = wh; i < W.size(); ++i) p1.push_back(W[i]);
    for (auto& s : A) p1.push_back(s);
    if (link(p1, ROOT + "\\data\\p1_final", "p1_final") < 0) return 1;

    // 退火：推理加浓，但保留 wiki/code 以免遗忘通用语言能力
    std::vector<std::string> p2;
    for (auto& s : R) p2.push_back(s);
    for (auto& s : A) p2.push_back(s);
    for (auto& s : R) p2.push_back(s);
    for (auto& s : A) p2.push_back(s);
    for (size_t i = 0; i < W.size() && i < 4; ++i) p2.push_back(W[i]);
    for (size_t i = 0; i < C.size() && i < 2; ++i) p2.push_back(C[i]);
    for (auto& s : R) p2.push_back(s);
    if (link(p2, ROOT + "\\data\\p1_anneal", "p1_anneal") < 0) return 1;
    return 0;
}

// ─────────────────────── 拒绝采样（取代 Python）───────────────────────
static int cmd_reject(const std::string& in, const std::string& out) {
    std::ifstream f(in, std::ios::binary);
    if (!f) { say("FAIL 无法读取轨迹 " + in); return 1; }
    std::ofstream o(out, std::ios::binary);
    if (!o) { say("FAIL 无法写入 " + out); return 1; }
    std::set<std::string> seen;
    std::string L; size_t n_in = 0, keep = 0, dup = 0;
    while (std::getline(f, L)) {
        if (!L.empty() && L.back()=='\r') L.pop_back();
        if (L.empty()) continue;
        ++n_in;
        std::vector<std::string> f5; size_t p = 0;
        for (int k = 0; k < 4; ++k) { size_t t = L.find('\t', p); if (t == std::string::npos) break; f5.push_back(L.substr(p, t - p)); p = t + 1; }
        if (f5.size() < 4) continue;
        f5.push_back(L.substr(p));
        if (f5[3] != "1") continue;                      // 必须判对
        if (f5[4].find("\xE6\x80\x9D\xE8\x80\x83") == std::string::npos) continue; // 必须含"思考"
        if (f5[4].empty()) continue;
        std::string key = f5[0] + "\x1f" + f5[4];
        if (!seen.insert(key).second) { ++dup; continue; }
        o << "DOC\nU " << f5[0] << "\nA " << f5[4] << "\n";
        ++keep;
    }
    o.flush();
    say("REJECT 输入=" + std::to_string(n_in) + " 保留=" + std::to_string(keep) +
        " 去重=" + std::to_string(dup) + " -> " + out);
    return keep > 0 ? 0 : 1;
}

// ─────────────────────── 训练阶段 ───────────────────────
static void base_env() {
    env("TAO_ALLOW_TOKENIZER", "1");
    env("TAO_CPU_THREADS", "16");
    env("TAO_CORPUS_THREADS", "16");
    // 架构硬约束：m==d，dk<=d（违反会被 Config::validate 拒绝）。
    // 规模对标 GPT-1：L=2 d=3200 s=1600 m=3200 dk=400 => 119.03M 参数。
    // L 固定为 2：层数会线性拖慢 CPU 解码，且 CPU 是唯一的解码执行者。
    // 权重与优化器状态整体驻主机 RAM（cudaHostAlloc Mapped），不占显存。
    env("TAO_CFG_LAYERS", "2"); env("TAO_CFG_D", "3200"); env("TAO_CFG_S", "1600");
    env("TAO_CFG_M", "3200"); env("TAO_CFG_DK", "400");
    env("TAO_OPT_OFFLOAD", "1");
    // 留出集：不设则收敛判据用训练 NLL，必然收敛到过拟合
    env("TAO_HOLDOUT", "128");
    env("TAO_HOLDOUT_TOKENS", "64");
}

static std::string trainer(const char* n) { return ROOT + "\\build\\" + n; }

// 批形状。GEMM 的 batch 等于 slots，width 是串行展开的时间步；显存正比
// slots*width（d=3200 实测约 113~129 MiB 每单位）。
// 【关键】峰值显存必须稳在 8188 MiB 之下留出余量：一旦逼近上限，WDDM 在每次
// 提交时做分配驱逐，ms_graph 从 ~1000ms 涨到 ~5000ms，**所有阶段均匀慢 3 倍**。
// 实测 d=3200：48 单位（16x3）峰值 6817 MiB 正常；64 单位（8x8）峰值 7776 MiB
// 触发 3 倍降速。因此默认 16x3=48 单位。
// 可用 TAO_PIPE_SLOTS / TAO_PIPE_WIDTH 覆盖而不必重新编译。
static unsigned g_slots() {
    if (const char* e = std::getenv("TAO_PIPE_SLOTS")) { const int v = std::atoi(e); if (v >= 1 && v <= 256) return unsigned(v); }
    return 16u;
}
static unsigned g_width() {
    if (const char* e = std::getenv("TAO_PIPE_WIDTH")) { const int v = std::atoi(e); if (v >= 1 && v <= 4096) return unsigned(v); }
    return 3u;
}
static std::string g_shape() { return std::to_string(g_slots()) + " " + std::to_string(g_width()); }

// 从训练日志解析最后完成的分片号（train_shards 每片结束会打印 SHARD_DONE index=N）。
// 用途：长训被中断后，从"下一片"继续，而不是把已训练过的分片再背一遍。
// 找不到时返回 -1，表示应当从第 0 片开始。
static int last_done_shard(const std::string& logpath) {
    std::ifstream f(logpath, std::ios::binary);
    if (!f) return -1;
    std::string all((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    int last = -1;
    const std::string key = "SHARD_DONE index=";
    size_t p = 0;
    while ((p = all.find(key, p)) != std::string::npos) {
        p += key.size();
        int v = std::atoi(all.c_str() + p);
        if (v > last) last = v;
    }
    return last;
}

// 组装 train_shards 的可选续训参数。
// 约定：RESUME_DIR 与 START_SHARD 必须成对出现（START_SHARD 是 argv[8]，
// 不传 RESUME_DIR 就无法传它），故这里统一构造，避免漏参导致静默从零开始。
static std::string resume_args(const std::string& outDir, const std::string& fallbackDir) {
    const std::string own = outDir + "\\opt_state.bin";
    std::ifstream f(own, std::ios::binary);
    if (f) {
        f.close();
        // 自身有断点：接着自己的进度跑
        int last = last_done_shard(outDir + ".out.log");
        return " " + q(outDir) + " " + std::to_string(last + 1);
    }
    if (!fallbackDir.empty()) {
        // 自身无断点，但需要从上游阶段接续权重（如退火必须接预训练）
        return " " + q(fallbackDir);
    }
    return std::string();
}

static int stage_pretrain() {
    if (exists(ROOT + "\\build\\L1_pretrain\\final.dsb")) { say("SKIP 预训练已完成"); return 0; }
    env("TAO_LR", "0.0006");
    // 长训口径：按实测 ~150 tok/s、语料 121M token 排 LR 表。
    // 收敛门控会提前结束，LR 未走到 MIN 是接受的结果（宁欠不溢）。
    env("TAO_LR_DECAY_START", "3000");
    env("TAO_LR_DECAY_STEPS", "300000");
    env("TAO_LR_MIN", "0.00003");
    env("TAO_WARMUP", "200");
    env("TAO_SHUFFLE_SEED", "20261101");
    env("TAO_CONVERGE", "1");
    env("TAO_CONV_WINDOW", "150"); env("TAO_CONV_Z", "2.0"); env("TAO_CONV_TOL", "0.0015");
    env("TAO_CONV_PATIENCE", "3"); env("TAO_CONV_MIN", "400"); env("TAO_CONV_FLOOR", "0.2");
    // 预训练是链条的起点，无上游可接；但自身被中断时必须能续跑 ——
    // 否则一次中断就丢掉之前全部分片的训练成果（长训下这是最贵的失败模式）。
    const std::string l1 = ROOT + "\\build\\L1_pretrain";
    const std::string rargs = resume_args(l1, "");
    if (!rargs.empty()) say("PRETRAIN 断点续跑" + rargs);
    std::string cmd = q(trainer("train_shards.exe")) +
        " " + q(ROOT + "\\data\\p1_final") + " " + q(ROOT + "\\build\\tok_v2.bbp") +
        " " + q(l1) + " 32000 " + g_shape() + rargs;
    int rc = run_proc(cmd, ROOT + "\\build\\L1_pretrain.out.log");
    if (rc != 0 || !exists(ROOT + "\\build\\L1_pretrain\\final.dsb")) { say("FAIL 预训练未产出"); return rc ? rc : 1; }
    return 0;
}

static int stage_anneal() {
    if (exists(ROOT + "\\build\\L2_anneal\\final.dsb")) { say("SKIP 退火已完成"); return 0; }
    env("TAO_LR", "0.0001");
    env("TAO_LR_DECAY_START", "0"); env("TAO_LR_DECAY_STEPS", "37200"); env("TAO_LR_MIN", "0.00001");
    env("TAO_WARMUP", "50");
    env("TAO_SHUFFLE_SEED", "20261201");
    unenv("TAO_CONVERGE");
    // 【关键】退火必须从预训练权重接续。原先此处只传了 6 个参数，
    // resume_dir 落到 argc<8 的默认空串 -> 退火从 initialize(seed) 的**随机初始化**开始，
    // 把预训练成果整段丢弃。链条 ②->③ 因此从未真正连通。
    // 现在：自身有断点则接自己，否则接 L1_pretrain。
    const std::string l2 = ROOT + "\\build\\L2_anneal";
    const std::string l1p = ROOT + "\\build\\L1_pretrain";
    const std::string rargs = resume_args(l2, l1p);
    say("ANNEAL 续训来源" + (rargs.empty() ? std::string("(无)") : rargs));
    std::string cmd = q(trainer("train_shards.exe")) +
        " " + q(ROOT + "\\data\\p1_anneal") + " " + q(ROOT + "\\build\\tok_v2.bbp") +
        " " + q(l2) + " 1200 " + g_shape() + rargs;
    int rc = run_proc(cmd, ROOT + "\\build\\L2_anneal.out.log");
    if (rc != 0 || !exists(ROOT + "\\build\\L2_anneal\\final.dsb")) { say("FAIL 退火未产出"); return rc ? rc : 1; }
    return 0;
}

static int stage_sft(const std::string& data, const std::string& out, const std::string& resume,
                     const std::string& updates, const std::string& lr, const char* tag) {
    if (exists(out + "\\final.dsb")) { say(std::string("SKIP ") + tag + " 已完成"); return 0; }
    env("TAO_FRESH", "0");
    env("TAO_WARMUP", "50");
    unenv("TAO_LR"); unenv("TAO_LR_MIN");  // SFT 的 LR 来自命令行参数，避免遗留值干扰
    std::string cmd = q(trainer("train_sft.exe")) + " " + q(data) + " " + q(ROOT + "\\build\\tok_v2.bbp") +
        " " + q(out) + " " + updates + " " + q(resume) + " " + g_shape() + " " + lr;
    int rc = run_proc(cmd, ROOT + "\\build\\" + tag + ".out.log");
    if (rc != 0 || !exists(out + "\\final.dsb")) { say(std::string("FAIL ") + tag + " 未产出"); return rc ? rc : 1; }
    return 0;
}

static int rollout(const std::string& model, const std::string& data, const std::string& extra,
                   const std::string& log) {
    std::string cmd = q(trainer("grpo_rollout.exe")) + " " + q(model) + " " + q(data) + " " + extra;
    return run_proc(cmd, log);
}

static int stage_r2() {
    const std::string base = ROOT + "\\build\\L3_sft";
    if (!exists(base + "\\final.dsb")) { say("FAIL 缺少 R1 SFT 产物"); return 1; }
    if (exists(ROOT + "\\build\\L4_grpo_r3\\final.dsb")) { say("SKIP R2 已完成"); return 0; }
    std::string common = "--n 60 --samples 16 --temp 1.1 --top-p 0.98 --max 512 --w-think 0.2";
    // --ref：冻结参考策略，论文 Eq.3 的 KL(π_theta||π_ref) 需要它。取 R1 冷启动 SFT 模型，
    // 且在 R2 各轮之间**保持不变** —— 参考策略若随训练漂移，KL 就失去锚点、约束失效。
    // 环境变量 TAO_GRPO_REF 可覆盖（用于消融：不设 --ref 时 KL 退化为 0）。
    std::string refArg;
    if (const char* rv = std::getenv("TAO_GRPO_REF")) { if (*rv) refArg = std::string(" --ref ") + q(rv); }
    else refArg = std::string(" --ref ") + q(base + "\\final.dsb");
    // 信号探测：组内方差为 0 的题不产生梯度，先量化再决定是否值得训练
    rollout(base + "\\final.dsb", ROOT + "\\data\\grpo_train.jsonl",
            common + refArg + " --dump " + q(ROOT + "\\build\\eval\\r2_signal.tsv"),
            ROOT + "\\build\\eval\\r2_signal.log");
    std::string src = base;
    for (int r = 1; r <= 3; ++r) {
        std::string dst = ROOT + "\\build\\L4_grpo_r" + std::to_string(r);
        std::string traj = ROOT + "\\build\\eval\\r2_r" + std::to_string(r) + ".tsv";
        rollout(src + "\\final.dsb", ROOT + "\\data\\grpo_train.jsonl",
                common + refArg + " --dump " + q(traj), ROOT + "\\build\\eval\\r2_r" + std::to_string(r) + "_roll.log");
        if (!exists(traj) || fsize(traj) < 64) { say("WARN R2 第 " + std::to_string(r) + " 轮无有效轨迹，跳过"); continue; }
        // train_grpo 会优先读 TAO_LR 环境变量（覆盖命令行参数），必须在此显式设定，
        // 否则会沿用退火阶段遗留的 1e-4，比预期高 5 倍。
        env("TAO_LR", "2e-5"); env("TAO_LR_MIN", "2e-6"); env("TAO_WARMUP", "20");
        std::string cmd = q(trainer("train_grpo.exe")) + " " + q(traj) + " " + q(ROOT + "\\build\\tok_v2.bbp") +
            " " + q(dst) + " 1200 " + q(src) + " " + g_shape() + " 2e-5";
        int rc = run_proc(cmd, ROOT + "\\build\\eval\\r2_r" + std::to_string(r) + "_train.log");
        if (rc == 0 && exists(dst + "\\final.dsb")) { say("R2 第 " + std::to_string(r) + " 轮完成 -> " + dst); src = dst; }
        else say("WARN R2 第 " + std::to_string(r) + " 轮未产出");
    }
    env("R2OUT", src);
    return 0;
}

static std::string r2out() { const char* v = std::getenv("R2OUT"); return v && *v ? v : (ROOT + "\\build\\L3_sft"); }

static int stage_r3() {
    // 必须用 R2 最终轮的轨迹，而不是 SFT 基线模型的轨迹
    std::string traj;
    for (int r = 3; r >= 1; --r) {
        std::string t = ROOT + "\\build\\eval\\r2_r" + std::to_string(r) + ".tsv";
        if (exists(t) && fsize(t) > 64) { traj = t; break; }
    }
    if (traj.empty()) { say("WARN R3 缺少 R2 轨迹，跳过"); return 0; }
    std::string dat = ROOT + "\\data\\r3_reject.txt";
    if (cmd_reject(traj, dat) != 0) { say("WARN R3 拒绝采样样本为空，跳过"); return 0; }
    return stage_sft(dat, ROOT + "\\build\\L5_reject_sft", r2out(), "4000", "2e-5", "L5_reject_sft");
}

static int stage_r4() {
    const std::string in = exists(ROOT + "\\build\\L5_reject_sft\\final.dsb")
        ? (ROOT + "\\build\\L5_reject_sft") : r2out();
    if (exists(ROOT + "\\build\\L6_align_r2\\final.dsb")) { say("SKIP R4 已完成"); return 0; }
    // 通用对齐无标准答案：奖励 = 充分性(0.3) + 非退化(0.5)
    std::string gen = "--n 80 --samples 8 --temp 1.0 --top-p 0.95 --max 384"
                      " --w-correct 0 --w-fmt 0 --w-think 0 --w-len 0.3 --w-norep 0.5";
    // R4 的参考策略取进入 R4 时的模型（R3 拒绝采样 SFT 产物），同样在各轮之间固定。
    std::string genRef = std::string(" --ref ") + q(in + "\\final.dsb");
    std::string src = in;
    for (int r = 1; r <= 2; ++r) {
        std::string dst = ROOT + "\\build\\L6_align_r" + std::to_string(r);
        std::string traj = ROOT + "\\build\\eval\\r4_r" + std::to_string(r) + ".tsv";
        rollout(src + "\\final.dsb", ROOT + "\\data\\general_rl.jsonl",
                gen + genRef + " --dump " + q(traj), ROOT + "\\build\\eval\\r4_r" + std::to_string(r) + "_roll.log");
        if (!exists(traj) || fsize(traj) < 64) { say("WARN R4 第 " + std::to_string(r) + " 轮无有效轨迹，跳过"); continue; }
        env("TAO_LR", "1e-5"); env("TAO_LR_MIN", "1e-6"); env("TAO_WARMUP", "20");
        std::string cmd = q(trainer("train_grpo.exe")) + " " + q(traj) + " " + q(ROOT + "\\build\\tok_v2.bbp") +
            " " + q(dst) + " 1000 " + q(src) + " " + g_shape() + " 1e-5";
        int rc = run_proc(cmd, ROOT + "\\build\\eval\\r4_r" + std::to_string(r) + "_train.log");
        if (rc == 0 && exists(dst + "\\final.dsb")) { say("R4 第 " + std::to_string(r) + " 轮完成 -> " + dst); src = dst; }
        else say("WARN R4 第 " + std::to_string(r) + " 轮未产出");
    }
    return 0;
}

static int stage_export() {
    std::string final_model[] = {
        ROOT + "\\build\\L6_align_r2", ROOT + "\\build\\L5_reject_sft", r2out()
    };
    std::string pick;
    for (auto& c : final_model) if (exists(c + "\\final.dsb")) { pick = c; break; }
    if (pick.empty()) { say("FAIL 无可导出的模型"); return 1; }
    fs::create_directories(ROOT + "\\build\\deploy");
    std::error_code ec;
    fs::copy_file(pick + "\\final.dsb", ROOT + "\\build\\deploy\\taovm_deploy.dsb",
                  fs::copy_options::overwrite_existing, ec);
    fs::copy_file(ROOT + "\\build\\tok_v2.bbp", ROOT + "\\build\\deploy\\taovm_tokenizer.bbp",
                  fs::copy_options::overwrite_existing, ec);
    say("EXPORT 源=" + pick + " -> build/deploy/taovm_deploy.dsb");
    // 真实加载验证：导出件必须能被独立解码器跑起来，否则导出无意义
    int rc = rollout(ROOT + "\\build\\deploy\\taovm_deploy.dsb", ROOT + "\\data\\grpo_test.jsonl",
                     "--n 20 --samples 1 --temp 0.7 --max 256 --w-think 0.2 --show 3",
                     ROOT + "\\build\\eval\\deploy_verify.log");
    if (rc != 0) { say("FAIL 部署验证：导出模型无法解码"); return rc; }
    say("OK 部署验证通过");
    return 0;
}

static int stage_eval(const std::string& model, const std::string& log) {
    return rollout(model, ROOT + "\\data\\grpo_test.jsonl",
                   "--n 200 --samples 4 --temp 0.7 --top-p 0.95 --max 512 --w-think 0.2", log);
}

// ─────────────────────── 主流程 ───────────────────────
static int cmd_all() {
    base_env();
    say("=========== 阶段① 数据工程 ===========");
    if (cmd_decontam() != 0) { say("FAIL 去污染失败"); return 1; }
    if (cmd_corpus_stage() != 0) { say("FAIL 分片失败"); return 1; }
    if (cmd_assemble() != 0) { say("FAIL 组装失败"); return 1; }

    say("=========== 阶段② 预训练 Base ===========");
    if (stage_pretrain() != 0) return 1;
    say("=========== 阶段③ 中期训练/退火 ===========");
    if (stage_anneal() != 0) return 1;

    say("=========== R1 冷启动 SFT ===========");
    if (stage_sft(ROOT + "\\data\\p1_reason_clean.txt", ROOT + "\\build\\L3_sft",
                  ROOT + "\\build\\L2_anneal", "12000", "5e-5", "L3_sft") != 0) return 1;

    say("=========== 阶段⑥a 评测：SFT 基线 ===========");
    stage_eval(ROOT + "\\build\\L3_sft\\final.dsb", ROOT + "\\build\\eval\\L3_sft_test.log");

    say("=========== R2 推理 RL（GRPO）===========");
    if (stage_r2() != 0) return 1;

    say("=========== R3 拒绝采样 + SFT ===========");
    stage_r3();

    say("=========== R4 全场景 RL ===========");
    stage_r4();

    say("=========== 阶段⑥b 最终评测 ===========");
    {
        std::string fin = exists(ROOT + "\\build\\L6_align_r2\\final.dsb") ? (ROOT + "\\build\\L6_align_r2")
                        : (exists(ROOT + "\\build\\L5_reject_sft\\final.dsb") ? (ROOT + "\\build\\L5_reject_sft") : r2out());
        stage_eval(fin + "\\final.dsb", ROOT + "\\build\\eval\\final_test.log");
    }

    say("=========== 阶段⑦⑧ 导出 + 部署验证 ===========");
    if (stage_export() != 0) return 1;

    say("PIPELINE_DONE");
    return 0;
}

int main(int argc, char** argv) {
    std::string cmd = argc > 1 ? argv[1] : "all";
    fs::create_directories(ROOT + "\\build");
    std::string logp = ROOT + "\\build\\pipeline.log";
    g_log = std::fopen(logp.c_str(), "a");
    say("================ taovm_pipe " + cmd + " ================");
    int rc = 2;
    try {
        if (cmd == "all") rc = cmd_all();
        else if (cmd == "decontam") rc = cmd_decontam();
        else if (cmd == "assemble") rc = cmd_assemble();
        else if (cmd == "corpus") rc = cmd_corpus_stage();
        else if (cmd == "reject") { if (argc < 4) { say("usage: reject IN.tsv OUT.txt"); rc = 2; } else rc = cmd_reject(argv[2], argv[3]); }
        else if (cmd == "r2") { base_env(); rc = stage_r2(); }
        else if (cmd == "r3") { base_env(); rc = stage_r3(); }
        else if (cmd == "r4") { base_env(); rc = stage_r4(); }
        else if (cmd == "pretrain") { base_env(); rc = stage_pretrain(); }
        else if (cmd == "anneal") { base_env(); rc = stage_anneal(); }
        else if (cmd == "export") { base_env(); rc = stage_export(); }
        else { say("未知命令 " + cmd); rc = 2; }
    } catch (const std::exception& e) {
        say(std::string("FATAL ") + e.what()); rc = 1;
    }
    say("taovm_pipe 结束 rc=" + std::to_string(rc));
    if (g_log) std::fclose(g_log);
    return rc;
}
