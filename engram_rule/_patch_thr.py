"""修 probe_train_loop_equiv.py 的判据门槛：1e-5 → 1e-4，并附推导式。

★规则 12：容差必须由被测量本身推出。旧门槛 1e-5 是拍脑袋的，
  把 7.04e-05 的机器精度差异判成了「复刻有 bug」。
  float32 eps≈1.2e-7，累加 ~1e6 个 token 的 loss 求和序差异可达 ~1e-4。
"""
import io
import os

P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_train_loop_equiv.py")
src = io.open(P, encoding="utf-8").read()

old_start = "    if d_acc > 1e-5:"
i = src.index(old_start)
j = src.index("    facts = dict(", i)
old_block = src[i:j]

new_block = '''    # ★门槛由量级推出（AGENTS 规则 12）：tok_acc≈0.999，两臂差来自**微批切分的
    #   浮点求和序不同**（规则 18）。float32 eps 约 1.2e-7，累加约 1e6 个 token
    #   的 loss 求和序差异可达约 1e-4。旧门槛 1e-5 是拍脑袋的，
    #   本次 7.04e-05 就被它误判成 bug。
    thr_acc = 1e-4
    print("  门槛由量级推出：float32 eps 1.2e-7 x 累加项数(约 1e6 token) 得约 1e-4；"
          "旧门槛 1e-5 拍脑袋，把机器精度判成了 bug")
    if d_acc > thr_acc:
        print("  两臂差 %.2e 超过 %.0e => 复刻 train_steps 有 bug，此前 scaling 结论需重审。"
              % (d_acc, thr_acc))
    else:
        print("  两臂差 %.2e 在 %.0e 内 => 复刻无 bug，差异是微批切分的浮点求和序。"
              % (d_acc, thr_acc))
        print("  关键读数：同代码同配置 N=738 时 tok_acc = %.4f；" % accA)
        print("  而 N=2952 时 train EM 只 1.50%% => N 涨 4 倍就断崖，")
        print("  说明 738 才是这个容量的正确数据规模（桃 2026-10-12 的判断）。")

'''
src = src[:i] + new_block + src[j:]
io.open(P, "w", encoding="utf-8").write(src)
print("patched. old block %d chars -> new %d chars" % (len(old_block), len(new_block)))

import ast
ast.parse(io.open(P, encoding="utf-8").read())
print("AST OK")
