# CSTR RL Control

自学强化学习过程中写的一个小项目，记录一下瞎折腾的过程，代码比较糙，欢迎指正。

## 这是什么

用 `gymnasium` 自己写了一个连续搅拌釜反应器（CSTR，Continuous Stirred-Tank Reactor）的简化仿真环境：

- 反应：A → B
- 状态：`[C_A, C_B, T]`（A浓度、B浓度、反应器温度）
- 动作：夹套温度的归一化控制量（-1 ~ 1）
- 目标：让 B 的浓度尽量稳定在设定值（0.6 mol/L）附近

然后用 `stable-baselines3` 的 SAC 算法去训练一个控制策略，让它自己学会怎么调夹套温度来把浓度控制住。

## 文件

- `cstr.py` — 环境定义 + 训练/测试脚本
- `sac_cstr.zip` — 训练好的 SAC 模型权重

## 怎么跑

```bash
pip install gymnasium stable-baselines3 numpy matplotlib

# 训练
python -c "from cstr import train_sac; train_sac()"

# 用训练好的模型测试并画图
python cstr.py
```

## 说明

纯粹是自学 RL 时候练手用的，环境和奖励函数都是简化过的玩具版本，没有做严谨的过程控制建模，别当真实工业场景参考。
