# Task 4: System Design - The Dynamic-Threshold Percent Model (DTPM)

## 1. 核心问题诊断
基于前三问的数据挖掘，现有赛制存在三大核心漏洞：
- **过度混乱 (Chaos Overload)**: 历史数据显示，**56.5%** 的淘汰者并非当周评委打分最低的选手。这意味着过半数的淘汰结果与专业表现不符。
- **才不配位 (Skill-Outcome Mismatch)**: 技能与最终排名的 Kendall相关系数仅为 **0.62**，说明"网红效应"严重干扰了比赛公正性。
- **盲投失效 (Blind Voting Failure)**: 粉丝投票在引入实时投票后，与当周表现几乎完全脱钩 (Correlation < 0.02)。

## 2. 新赛制设计理念：DTPM 模型

我们提出 **动态门槛百分比模型 (Dynamic-Threshold Percent Model, DTPM)**，由两大支柱构成：

### A. 动态权重 (Dynamic Weighting)
随着赛季深入，从"精英筛选"平滑过渡到"大众娱乐"。
$$ w_{judge}(t) = 0.90 - \frac{t-1}{T-1} \times (0.90 - 0.60) $$
- **初期 (Week 1)**: 评委 90% 权重。确保不会跳舞的选手被快速淘汰。
- **决赛 (Final)**: 评委 60% 权重。给粉丝足够的决定权，但仍保留专业底线。

### B. 表现门槛乘数 (Performance Gated Multiplier)
针对"盲投"和"才不配位"的熔断机制。
$$ \text{FanScore}_{adjusted} = \text{FanScore}_{raw} \times \gamma $$
$$ \gamma = \begin{cases} 1.0, & \text{if JudgeScore} \ge \text{Average} \\ 0.40, & \text{if JudgeScore} < \text{Average} \end{cases} $$
- 如果选手表现低于平均水准，其粉丝投票的效力将被打 **4折**。
- 这迫使流量明星必须努力达到及格线，否则再多的粉丝也救不了。

## 3. 仿真结果与性能对比

通过对 34 个历史赛季数据的 100% 重演模拟，DTPM 取得了压倒性的性能提升：

| 评估指标 | 历史赛制 (Baseline) | 新赛制 (DTPM) | 提升幅度 |
| :--- | :--- | :--- | :--- |
| **公平性 (Kendall's Tau)** | 0.6248 | **0.7942** | **+27.1%** |
| **逆转率 (Upset Rate)** | 56.47% | **14.93%** | **-73.6%** |

### 结果解读
- **黄金逆转区间**: 我们并没有将逆转率降为 0。14.9% 的逆转率意味着每 7 次淘汰中仍会有 1 次"爆冷" (Underdog Save)，这保留了真人秀所需的悬念和戏剧性，但剔除了恶性的"才不配位"。
- **专业与娱乐的平衡**: 帕累托优化前沿显示，$(w_{start}=0.9, w_{end}=0.6, \beta=0.4)$ 是现有权衡下的数学最优解。

## 4. 给制作人的建议 (Executive Summary)

采用 DTPM 系统将带来以下直接收益：
1.  **杜绝"鲍比·波恩斯"现象**: 类似 S27 冠军那样依靠粉丝票碾压专业的案例，在新赛制下将被 0.4 的惩罚系数直接熔断。
2.  **提升品牌公信力**: 公平性指标从 0.62 提升至 0.79，让"舞蹈"重新成为《与星共舞》的核心。
3.  **保留收视悬念**: 15% 的设计逆转率确保了粉丝仍然感受到每一票的价值，比赛不会变成沉闷的专业课。

## 5. 模型文件
- **优化代码**: `src/system_design.py`
- **基准评估**: `src/evaluate_baseline.py`
- **帕累托图表**: `results/plots/system_design/system_optimization_pareto.png`
