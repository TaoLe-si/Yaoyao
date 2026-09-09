# 自动训练观察说明

当前配置：CUDA Graph训练，累计目标2000次优化器更新；从360步恢复，先到400步，此后每100步保存并验证。常规CPU对话生成关闭，固定验证NLL仍在CPU计算。

## 固定输出
- build/training.log：追加写入的训练、验证、进程启动记录。
- build/loss_curve.csv：timestamp,kind,step,nll,lr。train是当前批次更新前assistant监督NLL；validation是固定验证集NLL。两条曲线应分别绘制，不可混为一条loss曲线。不同训练批次难度不同，单步波动不是退化的充分证据。
- build/graph_run_state.json：控制器阶段及最近完成验证的检查点。step不是实时训练步数；实时更新看CSV或UPDATE日志。

## 实时观察（PowerShell）
```powershell
Get-Content D:\TaoVm\build\training.log -Tail 30 -Wait
```

创建build/STOP_TRAINING会请求安全停止：GPU训练完成当前优化器更新后保存；CPU验证若已运行则等待其完成。不要强杀训练器或删除旧检查点。停止后的实际检查点以SAVED日志为准，控制器状态可能尚未反映中间停止步。

每100步验证后沿用原规则：改善至少0.02重置stale，连续3次不足改善则学习率减半，最低0.00003。周期由10步改为100步后，触发回退所需训练步数相应增加。验证达到2.5时保存并进入质量审查，不等于能力目标完成。
