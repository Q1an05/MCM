# 完整建模思路文档

> **文档说明**：本文档整理了 MCM 2026 Problem C "与星共舞"（Dancing with the Stars，DWTS）比赛的完整建模思路。按照四个研究问题分别展开，包含详细的方法论、公式推导以及所有参数的物理意义解释。
>
> **题目背景**：美国电视节目《与星共舞》是一场名人与专业舞者搭档的舞蹈比赛。比赛结果由专业评委（1-10 分制的多支舞蹈评分之和）和观众投票共同决定。历史上节目组曾多次修改淘汰规则（排名制、百分比制、评委拯救制），以平衡"技术性"与"娱乐性"之间的张力。
>
> **核心挑战**：观众投票数据在播出前是保密的，我们只能观察到评委分数和最终淘汰结果。建模的核心任务是**反向推算**观众投票分布，并在此基础上进行规则对比、因素分析和赛制设计。
>
> **建模框架概览**：
> - **Q1**：贝叶斯-狄利克雷混合模型——从评委分和淘汰结果反向推算观众投票
> - **Q2**：静态反事实分析——对比三种规则系统对淘汰结果的影响差异
> - **Q3**：双轨归因模型 + K-Means 聚类——分析舞伴、行业、年龄等因素的效应
> - **Q4**：动态门槛百分比模型（DTPM）——设计更公平的新赛制并进行回测验证

---

## 目录

1. [问题定义与研究框架](#1-问题定义与研究框架)
2. [Question 1：贝叶斯-狄利克雷混合模型](#2-question-1贝叶斯-狄利克雷混合模型)
3. [Question 2：规则系统对比与反事实分析](#3-question-2规则系统对比与反事实分析)
4. [Question 3：因素分析与行业聚类](#4-question-3因素分析与行业聚类)
5. [Question 4：新赛制设计与验证](#5-question-4新赛制设计与验证)
6. [附录：参数符号表与指标定义](#6-附录参数符号表与指标定义)

---

## 1. 问题定义与研究框架

### 1.1 四个研究问题的逻辑关系

| 问题编号 | 研究问题 | 核心目标 | 建模方法 | 依赖关系 |
|---------|---------|---------|---------|---------|
| **Q1** | 观众投票分布是什么？ | 反向推算每周各选手的粉丝投票份额 | 贝叶斯-狄利克雷混合模型 | 基础输入 |
| **Q2** | 哪种规则更偏向粉丝投票？ | 对比三种规则系统的公平性差异 | 静态反事实分析 | 依赖 Q1 产出 |
| **Q3** | 哪些因素影响选手表现？ | 分析舞伴、行业、年龄等因素的效应 | 双轨归因模型 + K-Means 聚类 | 独立分析 |
| **Q4** | 如何设计更公平的赛制？ | 提出并验证新的淘汰机制 | DTPM 模型 + Pareto 优化 | 整合 Q1-Q3 结论 |

### 1.2 三种历史规则系统的定义

#### 规则 A：排名加和法（Rank System）

**适用赛季**：第 1-2 季、第 28 季及以后（部分）

**核心公式**：

$$\text{TotalRank}_i = \text{JudgeRank}_i + \text{FanRank}_i$$

**淘汰规则**：

$$\text{Loser} = \arg\max_i (\text{TotalRank}_i)$$

当出现平局时（TotalRank 相等），粉丝排名较差的选手被淘汰。

**参数说明**：

| 参数符号 | 物理意义 | 取值范围 | 说明 |
|---------|---------|---------|------|
| $\text{JudgeRank}_i$ | 选手 $i$ 的评委分排名 | $\{1, 2, ..., n\}$ | 1 表示最高评委分 |
| $\text{FanRank}_i$ | 选手 $i$ 的粉丝投票排名 | $\{1, 2, ..., n\}$ | 1 表示最多粉丝票 |
| $\text{TotalRank}_i$ | 选手 $i$ 的总排名得分 | $[2, 2n]$ | 两名次相加，无单位 |
| $n$ | 当周参赛选手总数 | 正整数 | 因周次而异 |

**特性分析**：排名是**序数型**数据，分差大小不重要，只看相对位置。这意味着 9.9 分和 9.0 分的差异，与 8.0 分和 7.1 分的差异在排名系统中"等效"（都是差 1 名）。

---

#### 规则 B：百分比加和法（Percent System）

**适用赛季**：第 3-27 季

**核心公式**：

$$\text{TotalScore}_i = \text{JudgeShare}_i + \text{FanShare}_i$$

其中各分项的计算方式为：

$$\text{JudgeShare}_i = \frac{\text{JudgeScore}_i}{\sum_{j=1}^{n} \text{JudgeScore}_j} \times 100\%$$

$$\text{FanShare}_i = \frac{\text{FanVotes}_i}{\sum_{j=1}^{n} \text{FanVotes}_j} \times 100\%$$

**淘汰规则**：

$$\text{Loser} = \arg\min_i (\text{TotalScore}_i)$$

**参数说明**：

| 参数符号 | 物理意义 | 取值范围 | 说明 |
|---------|---------|---------|------|
| $\text{JudgeScore}_i$ | 选手 $i$ 的评委总分 | $[0, \infty)$ | 多支舞蹈分数之和 |
| $\text{FanVotes}_i$ | 选手 $i$ 的观众投票数 | $[0, \infty)$ | 保密数据，需要反向推算 |
| $\text{JudgeShare}_i$ | 选手 $i$ 的评委分占比 | $[0, 1]$ | 归一化后的比例 |
| $\text{FanShare}_i$ | 选手 $i$ 的粉丝票占比 | $[0, 1]$ | 归一化后的比例 |
| $\text{TotalScore}_i$ | 选手 $i$ 的综合得分 | $[0, 2]$ | 两个百分比相加，单位为 100% |

**特性分析**：百分比是**基数型**数据，对分差敏感。极低评委分的选手需要巨量粉丝票才能弥补分数差距。这导致了一个"Bobby Bones 效应"——某些高人气低评委分的选手可以利用粉丝基数优势"碾压"专业评分。

---

#### 规则 C：排名+评委拯救制（Rank With Save）

**适用赛季**：第 28 季及以后

**操作步骤**：

1. 先按规则 A 计算 $\text{TotalRank}$，找出倒数两名（Bottom Two）
2. 评委在直播中投票决定 Bottom Two 中谁留下谁淘汰

**公式**：

$$P(\text{Saved}_i) = \begin{cases}
\text{MERIT\_SAVE\_PROB} & \text{if JudgeScore}_i > \text{JudgeScore}_j \\
1 - \text{MERIT\_SAVE\_PROB} & \text{if JudgeScore}_i < \text{JudgeScore}_j
\end{cases}$$

**参数说明**：

| 参数符号 | 物理意义 | 最优值 | 说明 |
|---------|---------|-------|------|
| $\text{MERIT\_SAVE\_PROB}$ | 评委理性拯救概率 | 0.775 | 基于历史数据分析得出 |
| $\text{JudgeScore}_i$ | Bottom Two 中较高分者的评委分 | 原始数据 | 用于比较 |
| $\text{JudgeScore}_j$ | Bottom Two 中较低分者的评委分 | 原始数据 | 用于比较 |
| $P(\text{Saved}_i)$ | 高分者被拯救的概率 | $[0, 1]$ | 条件概率 |

**核心洞察**：评委分数较高者有 77.5% 的概率被拯救，这保护了"高技术低人气"的选手免于被淘汰。

---

## 2. Question 1：贝叶斯-狄利克雷混合模型

### 2.1 问题定义与建模目标

**核心挑战**：观众投票数据是保密的，我们只能观察到评委分数和最终淘汰结果。建模目标是通过贝叶斯推断，从已知结果反向推算未知的观众投票分布。

**建模目标**：

1. **生成估算的观众投票份额**：为每周每位选手生成一个"估算的粉丝投票份额" $\hat{\vec{V}}_{\text{share}}$
2. **量化模型不确定性**：通过蒙特卡洛模拟，计算模型置信度和预测区间
3. **识别混沌因素比例**：量化投票结果中"可预测"与"不可预测"成分的比例

### 2.2 贝叶斯-狄利克雷双世界模型

#### 2.2.1 核心思想

将每周的投票结果解构为两个**正交分量**的混合：

1. **理性分量（Rational Component）**：由选手的长期历史表现（积累的技能评价 $\vec{\alpha}_{\text{skill}}$）和当周评委打分（Z-Score）共同驱动。这部分反映"技术导向"的投票行为。

2. **混沌分量（Chaotic Component）**：代表不可预测的社会动力学因素，如粉丝的非理性狂热、同情票效应、突发新闻影响等。这部分反映"娱乐导向"的投票行为。

#### 2.2.2 混合狄利克雷分布模型

**核心公式**：

$$\vec{V}_{\text{share}} \sim (1 - \lambda) \cdot \text{Dir}(\vec{\alpha}_{\text{skill}}) + \lambda \cdot \text{Dir}(\vec{\alpha}_{\text{chaos}})$$

这个公式表示最终的投票份额分布是**两个狄利克雷分布的凸组合**。

**参数说明**：

| 参数符号 | 物理意义 | 类型 | 说明 |
|---------|---------|------|------|
| $\vec{V}_{\text{share}}$ | 粉丝投票份额向量 | 随机向量 | $n$ 维，元素和为 1 |
| $\lambda$ | 混沌权重（待优化参数） | 标量 | $[0, 1]$，表示混沌成分的比例 |
| $\text{Dir}(\cdot)$ | 狄利克雷分布 | 概率分布 | 投票生成的概率模型 |
| $\vec{\alpha}_{\text{skill}}$ | 理性/技能驱动的先验参数 | 向量 | 由上周后验 + 当周评委分数更新 |
| $\vec{\alpha}_{\text{chaos}}$ | 混沌分布的先验参数 | 向量 | 由混沌分布类型决定 |

**狄利克雷分布的性质**：

- $\vec{\alpha} > 1$：分布呈"倒 U 型"，各选手份额相对均衡
- $\vec{\alpha} = 1$：均匀分布，各选手份额趋于相等
- $0 < \vec{\alpha} < 1$：分布呈"U 型"，份额高度不均

---

### 2.3 先验分布的跨周传递

#### 2.3.1 初始化先验

**公式**：

$$\vec{\alpha}^{(0)} = \text{initialize\_priors}(\text{contestant\_names})$$

**代码实现**：

```python
def initialize_priors(contestant_names: List[str]) -> np.ndarray:
    n = len(contestant_names)
    return np.full(n, INITIAL_ALPHA)  # INITIAL_ALPHA = 1.0
```

**参数说明**：

| 参数符号 | 物理意义 | 最优值 | 说明 |
|---------|---------|-------|------|
| $\vec{\alpha}^{(0)}$ | 第 1 周的初始先验向量 | $\vec{1}$ | 全部元素为 1 |
| $\text{INITIAL\_ALPHA}$ | 无信息先验参数 | 1.0 | 对应均匀分布倾向 |
| $n$ | 当周参赛选手数量 | 正整数 | 因周次而异 |

#### 2.3.2 跨周传递规则

**公式**：

$$\vec{\alpha}^{(t)} = \text{align\_prior\_to\_current}(\vec{\alpha}^{(t-1)}, \text{names}^{(t)})$$

**详细步骤**：

1. **继承上周后验**（继续参赛的选手）：
   $$\alpha_i^{(t)} = \alpha_i^{(t-1)}, \quad \text{if contestant } i \text{ continued}$$

2. **新选手初始化**：
   $$\alpha_i^{(t)} = \text{INITIAL\_ALPHA}, \quad \text{if contestant } i \text{ is new}$$

3. **缩放调整**（保持相对比例）：
   $$\text{scale\_factor} = \min\left(\frac{1}{\text{remaining\_ratio}}, 1.3\right)$$
   $$\vec{\alpha}^{(t)} = \vec{\alpha}^{(t)} \times \text{scale\_factor}$$
   $$\vec{\alpha}^{(t)} = \max(\vec{\alpha}^{(t)}, \text{MIN\_ALPHA})$$

**参数说明**：

| 参数符号 | 物理意义 | 取值 | 说明 |
|---------|---------|------|------|
| $\vec{\alpha}^{(t)}$ | 第 $t$ 周的先验向量 | 向量 | 模型状态变量 |
| $\vec{\alpha}^{(t-1)}$ | 第 $t-1$ 周的后验向量 | 向量 | 模型状态变量 |
| $\text{remaining\_ratio}$ | 选手保留比例 | $n^{(t)} / n^{(t-1)}$ | 衡量淘汰速度 |
| $\text{scale\_factor}$ | 缩放系数 | $[1, 1.3]$ | 防止 $\alpha$ 衰减过快 |
| $\text{MIN\_ALPHA}$ | 最小先验值 | 0.1 | 防止数值下溢 |

---

### 2.4 评委分数对先验的修正

#### 2.4.1 Z-Score 标准化

**公式**：

$$z_i = \frac{\text{JudgeScore}_i - \mu}{\sigma}$$

其中：

$$\mu = \frac{1}{n}\sum_{j=1}^{n} \text{JudgeScore}_j$$

$$\sigma = \sqrt{\frac{1}{n-1}\sum_{j=1}^{n}(\text{JudgeScore}_j - \mu)^2}$$

**参数说明**：

| 参数符号 | 物理意义 | 计算方式 | 说明 |
|---------|---------|---------|------|
| $\text{JudgeScore}_i$ | 选手 $i$ 当周评委总分 | 原始数据 | 多支舞蹈分数之和 |
| $\mu$ | 当周评委分的算术均值 | $\frac{1}{n}\sum \text{JudgeScore}_j$ | 衡量当周整体水平 |
| $\sigma$ | 当周评委分的标准差 | $\sqrt{\frac{1}{n-1}\sum(\cdot)^2}$ | 衡量分数离散程度 |
| $z_i$ | 选手 $i$ 的 Z-Score | $(JudgeScore_i - \mu) / \sigma$ | 标准化后的相对位置 |

#### 2.4.2 先验修正乘数

**公式**：

$$\text{multiplier}_i = 1 + \text{SKILL\_IMPACT\_FACTOR} \times z_i$$

$$\text{multiplier}_i = \text{clip}(\text{multiplier}_i, 0.5, 2.0)$$

$$\alpha_i^{(t)} = \alpha_i^{(t)} \times \text{multiplier}_i$$

**参数说明**：

| 参数符号 | 物理意义 | 取值 | 说明 |
|---------|---------|------|------|
| $\text{SKILL\_IMPACT\_FACTOR}$ | 技能影响力系数 | 0.3 | 控制评委分对先验的影响强度 |
| $\text{clip}(\cdot, 0.5, 2.0)$ | 截断函数 | — | 防止极端扭曲，保持模型稳定 |

**物理意义示例**：

- $z_i = 2$（极高评委分）：$\text{multiplier} = 1 + 0.3 \times 2 = 1.6$（α 增大 60%）
- $z_i = 0$（平均评委分）：$\text{multiplier} = 1 + 0.3 \times 0 = 1.0$（α 不变）
- $z_i = -2$（极低评委分）：$\text{multiplier} = 1 + 0.3 \times (-2) = 0.4$（α 减小 60%）

---

### 2.5 贝叶斯后验更新

#### 2.5.1 后验计算公式

**公式**：

$$\vec{\alpha}^{(t)}_{\text{posterior}} = \text{update\_prior\_with\_evidence}(\vec{\alpha}^{(t)}_{\text{prior}}, \hat{\vec{V}}_{\text{share}})$$

$$\vec{\alpha}^{(t)}_{\text{posterior}} = \vec{\alpha}^{(t)}_{\text{prior}} + \eta \times (\hat{\vec{V}}_{\text{share}} \times \text{EVIDENCE\_BOOST})$$

$$\vec{\alpha}^{(t)}_{\text{posterior}} = \max(\vec{\alpha}^{(t)}_{\text{posterior}}, \text{MIN\_ALPHA})$$

**参数说明**：

| 参数符号 | 物理意义 | 取值 | 说明 |
|---------|---------|------|------|
| $\eta$ | 学习率（LEARNING\_RATE） | 0.4 | 控制后验更新的步长 |
| $\hat{\vec{V}}_{\text{share}}$ | 估算的粉丝份额 | 向量 | 模型输出，用于更新先验 |
| $\text{EVIDENCE\_BOOST}$ | 证据增强系数 | 5.0 | 放大证据的影响，加速学习 |

---

### 2.6 混合模型：理性+混沌分量

#### 2.6.1 模拟采样公式

**公式**：

$$n_{\text{skill}} = N_{\text{total}} \times (1 - \lambda)$$

$$n_{\text{chaos}} = N_{\text{total}} \times \lambda$$

$$\text{samples}_{\text{skill}} \sim \text{Dirichlet}(\vec{\alpha}_{\text{skill}})$$

$$\text{samples}_{\text{chaos}} \sim \text{Dirichlet}(\vec{\alpha}_{\text{chaos}})$$

$$\text{samples} = \text{concat}(\text{samples}_{\text{skill}}, \text{samples}_{\text{chaos}})$$

**参数说明**：

| 参数符号 | 物理意义 | 取值 | 说明 |
|---------|---------|------|------|
| $N_{\text{total}}$ | 总模拟次数 | 10,000 | 蒙特卡洛模拟的采样数量 |
| $\lambda$ | 混沌比例 | 优化目标 | 最优值约为 0.024 |
| $\text{samples}_{\text{skill}}$ | 理性分量样本 | 矩阵 | $(1-\lambda) \times N_{\text{total}}$ 行 |
| $\text{samples}_{\text{chaos}}$ | 混沌分量样本 | 矩阵 | $\lambda \times N_{\text{total}}$ 行 |

#### 2.6.2 混沌分布类型

| 分布类型 | 先验参数 $\vec{\alpha}_{\text{chaos}}$ | 物理含义 | 优化结果 |
|---------|---------------------------------------|---------|---------|
| **Uniform** | $\vec{\alpha}_{\text{chaos}} = \vec{1}$ | 完全随机的噪声 | 备选 |
| **Pareto** | $\alpha_{\text{chaos},i} \propto \text{Pareto}(1.5)$ | 长尾分布，极端偏好弱者 | 备选 |
| **Exponential** | $\alpha_{\text{chaos},i} \propto \text{Exponential}(1.0)$ | 中等尾巴，温和的弱者偏好 | **最优** |

---

### 2.7 香农熵（不确定性度量）

#### 2.7.1 熵计算公式

**公式**：

$$H_i = -\sum_{k=1}^{50} p_k \ln(p_k)$$

**参数说明**：

| 参数符号 | 物理意义 | 取值 | 说明 |
|---------|---------|------|------|
| $H_i$ | 选手 $i$ 的香农熵 | 标量 | 单位：nats（自然对数单位） |
| $p_k$ | 选手 $i$ 在第 $k$ 个区间的投票份额概率 | $[0, 1]$ | 概率质量函数 |
| 50 | 区间划分数量 | 常量 | 将 $[0, 1]$ 划分为 50 个桶 |

**物理意义**：熵值越高表示模型对投票份额越不确定，即该选手的投票行为越难预测。

---

### 2.8 模型优化：信息比率（IR）

#### 2.8.1 评价指标定义

**解释率（Explanation Rate）**：

$$\text{Explanation Rate} = \frac{\text{成功解释的周数}}{\text{总消除周数}} \times 100\%$$

**置信度（Certainty）**：

$$\text{Certainty} = \frac{n_{\text{valid\_sims}}}{N_{\text{total}}}$$

#### 2.8.2 信息比率公式

**公式**：

$$\text{IR}(\lambda) = \frac{\text{Explanation Rate}}{1 - \text{Certainty}}$$

#### 2.8.3 优化结果

| 分布模型 | 最佳 $\lambda$ | Information Ratio | 解释率 | 置信度 |
|---------|---------------|-------------------|--------|--------|
| **Exponential** | **0.024** | **1.479** | **95.5%** | **35.4%** |
| Uniform | 0.097 | 1.477 | 97.7% | 33.8% |
| Pareto | 0.017 | 1.473 | 93.9% | 36.2% |

**核心结论**：

- **最优混沌分布**：Exponential（指数分布）
- **最优混沌权重**：$\lambda = 0.024$（2.4%）
- **物理意义**：投票结果由 **97.6% 理性因素** + **2.4% 混沌因素** 决定

---

## 3. Question 2：规则系统对比与反事实分析

### 3.1 问题定义与建模目标

**核心问题**：哪种投票规则更偏向粉丝投票？对不同类型选手（高技术/低人气、低技术/高人气）的影响有何差异？

**建模目标**：

1. 构建"多重宇宙模拟器"，对历史每周应用三种规则
2. 计算多维度公平性指标
3. 通过争议案例验证结论

### 3.2 方法论：静态反事实分析

#### 3.2.1 核心思想

**静态反事实分析**的要点：

- **输入固定**：保持评委分数和 Q1 估算的粉丝份额不变
- **规则切换**：对同一周应用三种不同的淘汰规则
- **结果对比**：观察"谁会被淘汰"
- **不动态推演**：不假设被淘汰选手在后续周次的虚构表现

### 3.3 三种规则的蒙特卡洛实现

#### 3.3.1 排名制（Rank System）

**公式**：

$$\text{FanRank}_i = \text{compute\_fan\_rank}(\text{FanShare}_i)$$

$$\text{TotalRank}_i = \text{JudgeRank}_i + \text{FanRank}_i$$

$$\text{Eliminated} = \arg\max_i(\text{TotalRank}_i)$$

**平局处理**：

$$\text{Eliminated} = \arg\max_{\text{TotalRank}_i = \text{TotalRank}_{\max}} (\text{FanRank}_i)$$

**参数说明**：

| 参数符号 | 物理意义 | 说明 |
|---------|---------|------|
| $\text{compute\_fan\_rank}$ | 排名计算函数 | 使用 `method='average'` 处理并列排名 |

#### 3.3.2 百分比制（Percent System）

**公式**：

$$\text{JudgeShare}_i = \frac{\text{JudgeScore}_i}{\sum_j \text{JudgeScore}_j}$$

$$\text{TotalShare}_i = \text{JudgeShare}_i + \text{FanShare}_i$$

$$\text{Eliminated} = \arg\min_i(\text{TotalShare}_i)$$

#### 3.3.3 排名+评委拯救制（Rank With Save）

**公式**：

1. 计算 $\text{TotalRank}_i = \text{JudgeRank}_i + \text{FanRank}_i$
2. 识别 Bottom Two（排名最高者 = 最差）
3. 评委拯救判定：

$$P(\text{Eliminated}_i) = \begin{cases}
1 - \text{MERIT\_SAVE\_PROB} & \text{if JudgeScore}_i < \text{JudgeScore}_j \\
\text{MERIT\_SAVE\_PROB} & \text{if JudgeScore}_i > \text{JudgeScore}_j
\end{cases}$$

---

### 3.4 量化评价指标体系

#### 3.4.1 逆转率（Reversal Rate）

**公式**：

$$\text{Reversal Rate} = \frac{\text{规则冲突的周数}}{\text{总周数}} \times 100\%$$

**结果**：13.43%（约 1/7 的周次在不同规则下产生不同结果）

---

#### 3.4.2 粉丝权重指数（Fan Power Index, FPI）

**公式**：

$$\text{FPI} = |\rho(\text{FinalRank}, \text{FanShare})|$$

其中 $\rho$ 是 Spearman 等级相关系数。

**结果对比**：

| 规则系统 | FPI | 解读 |
|---------|-----|------|
| Percent | **0.847** | 最高粉丝影响力 |
| Rank | 0.793 | 中等影响力 |
| Rank+Save | 0.793 | 中等影响力 |

---

#### 3.4.3 平庸存活率（Mediocrity Survival Rate）

**定义**：评委分数倒数前三名的选手在该规则下的存活率。

**公式**：

$$\text{Mediocrity Survival} = \frac{\text{低分选手存活数}}{\text{低分选手总数}} \times 100\%$$

**结果对比**：

| 规则系统 | 平庸存活率 | 解读 |
|---------|-----------|------|
| Percent | 71.09% | 最强"人气保护盾"，低分选手最不易被淘汰 |
| Rank | 68.30% | 中等保护 |
| **Rank+Save** | **67.40%** | **最有效淘汰低分选手** |

---

#### 3.4.4 英才被杀率（Talent Elimination Rate）

**定义**：评委分数前三名的选手被淘汰的比例。

**公式**：

$$\text{Talent Elimination} = \frac{\text{高分选手淘汰数}}{\text{高分选手总数}} \times 100\%$$

**结果对比**：

| 规则系统 | 英才被杀率 | 解读 |
|---------|-----------|------|
| Percent | 4.50% | 保护最差，高技术选手易被淘汰 |
| Rank | 3.20% | 中等保护 |
| **Rank+Save** | **2.00%** | **最强技术保护** |

---

### 3.5 争议案例验证

#### 3.5.1 案例汇总

| 选手 | 赛季 | Bottom 3 频率 | Rank 淘汰周 | Percent 淘汰周 | Save 淘汰周 | 实际成绩 |
|-----|------|--------------|------------|---------------|------------|---------|
| Jerry Rice | S2 | 75% (6/8) | W8 | W7 | W7 | 亚军 |
| Billy Ray Cyrus | S4 | 63% (5/8) | W1 | W1 | W1 | 第 5 名 |
| Bristol Palin | S11 | **80% (8/10)** | W5 | **W10** | W5 | 第 3 名 |
| Bobby Bones | S27 | 78% (7/9) | W9 | W9 | **W8** | **冠军** |

#### 3.5.2 核心洞察

1. **Bristol Palin 是 Percent 最大受益者**：实际第 10 周存活（最终第 3 名），而 Rank/Save 第 5 周淘汰，证明 Percent 系统对低技术高人气选手的极端保护。

2. **Bobby Bones 验证 Save 机制有效性**：在 Save 下提前 1 周被淘汰，印证 2018 年规则改革的必要性。

3. **Billy Ray Cyrus 三系统一致性**：技术-人气双低导致第 1 周即被淘汰，规则影响有限。

---

## 4. Question 3：因素分析与行业聚类

### 4.1 问题定义与建模目标

**核心问题**：哪些因素影响选手表现？专业舞伴、明星特征（年龄、行业）如何影响技术评分和人气投票？

**建模目标**：

1. 构建双轨归因模型，分别分析技术轨道和人气轨道
2. 通过 K-Means 聚类对 26 种职业进行降维
3. 验证模型假设（年龄线性、成长轨迹一致性）
4. 分析盲投效应和舞伴效应

### 4.2 双轨归因模型（Dual-Track Attribution Model）

#### 4.2.1 模型架构

将选手表现分解为两个**独立赛道**：

| 赛道 | 因变量 | 衡量内容 | 模型代号 |
|-----|-------|---------|---------|
| **技术轨道** | $\text{Score}_{\text{normalized}}$ | 评委评分（舞蹈技术水平） | Model A |
| **人气轨道** | $\log(\text{FanShare})$ | 粉丝投票份额（观众喜爱度） | Model B |

---

### 4.3 行业聚类（K-Means Clustering）

#### 4.3.1 三维评分体系

**公式**：

$$\text{IndustryScore}_i = (\text{Physicality}_i, \text{Performance}_i, \text{Fanbase}_i)$$

**参数说明**：

| 参数符号 | 物理意义 | 取值范围 | 说明 |
|---------|---------|---------|------|
| $\text{Physicality}_i$ | 行业 $i$ 的体能要求 | 1-5 | 1=最低（演员），5=最高（运动员） |
| $\text{Performance}_i$ | 行业 $i$ 的舞台表现要求 | 1-5 | 1=最低，5=最高（歌手） |
| $\text{Fanbase}_i$ | 行业 $i$ 的固有粉丝基础 | 1-5 | 1=最低，5=最高（真人秀明星） |

#### 4.3.2 最优 K 值选择

**轮廓系数公式**：

$$s = \frac{b - a}{\max(a, b)}$$

其中：
- $a$：簇内平均距离（样本到同簇其他样本的平均距离）
- $b$：最近簇的平均距离（样本到最近其他簇的平均距离）

**K 值选择结果**：

| K | Inertia (SSE) | Silhouette Score | 推荐 |
|---|--------------|-----------------|------|
| 2 | 48.91 | 0.4067 | |
| **3** | **31.81** | **0.4595** | **✓ 最优** |
| 4 | 19.44 | 0.4472 | |
| 5 | 15.99 | 0.3916 | |

#### 4.3.3 聚类结果

| 簇名称 | Physicality | Performance | Fanbase | 样本占比 | 典型职业 |
|-------|------------|------------|---------|---------|---------|
| **Athletic Elite** | 4.17 | 2.00 | 2.83 | 24.6% | 运动员、奥运冠军、赛车手 |
| **Performance Artist** | 1.53 | 3.82 | 2.47 | 55.7% | 演员、歌手、喜剧演员 |
| **Fan Favorite** | 1.00 | 3.00 | 5.00 | 19.7% | 真人秀明星、网红、选美冠军 |

---

### 4.4 混合效应模型（Linear Mixed Model, LMM）

#### 4.4.1 技术轨道（Model A）

**公式**：

$$\text{Score}_i = \beta_0 + \beta_1 \text{Cluster}_i + \beta_2 \text{Week}_i + \beta_3 (\text{Cluster} \times \text{Week})_i + \beta_4 \text{Age}_i + v_{\text{Partner}_i} + \epsilon_i$$

**参数估计结果**：

| 参数 | 定义 | 系数 | Std.Err | z 值 | P 值 | 显著性 |
|-----|-----|------|--------|------|------|-------|
| $\beta_0$ | Athletic Elite 基线 | 0.637 | 0.009 | 69.59 | <0.001 | *** |
| $\beta_1(\text{Fan Favorite})$ | Fan Favorite 效应 | -0.048 | 0.011 | -4.33 | <0.001 | *** |
| $\beta_1(\text{Performance Artist})$ | Performance Artist 效应 | 0.007 | 0.009 | 0.78 | 0.438 | |
| $\beta_2$ | 每周成长 | +0.030 | 0.001 | 22.76 | <0.001 | *** |
| $\beta_3(\text{Fan Favorite} \times \text{Week})$ | 追赶效应 | +0.004 | 0.002 | 2.11 | 0.035 | * |
| $\beta_4$ | 年龄效应（标准化） | -0.044 | 0.002 | -19.79 | <0.001 | *** |
| $v_{\text{Partner}}$ | 舞伴随机效应 | $\sigma^2 = 0.001$ | | | | |
| $\epsilon_i$ | 残差 | $\sigma^2 = 0.0092$ | | | | |

**参数说明**：

| 参数符号 | 物理意义 | 说明 |
|---------|---------|------|
| $\beta_0$ | Athletic Elite 聚类的基线技术分 | 控制组的平均表现 |
| $\beta_1(\text{Fan Favorite})$ | Fan Favorite 相对于 Athletic Elite 的技术劣势 | 负值表示起点更低 |
| $\beta_2$ | 每周技术提升幅度 | 所有选手平均每周进步 3% |
| $\beta_3(\text{Fan Favorite} \times \text{Week})$ | Fan Favorite 的追赶速度 | 额外 0.4%/周 |
| $\beta_4$ | 年龄对技术的负面影响 | 年龄每增加 1 个标准差，技术分下降 4.4% |
| $v_{\text{Partner}_i}$ | 舞伴 $i$ 的随机效应 | 控制舞伴质量差异 |

**关键发现**：

- 🏋️ **运动员优势**：Athletic Elite 起点最高（基线）
- 📉 **粉丝型劣势**：Fan Favorite 技术评分低 4.8%
- 📈 **成长效应**：每周技术提升 3%
- 🚀 **追赶效应**：Fan Favorite 虽然起点低，但成长更快
- 👴 **年龄惩罚**：年龄每增加 1 个标准差，技术评分下降 4.4%

---

#### 4.4.2 人气轨道（Model B）

**公式**：

$$\log(\text{FanShare}_i) = \alpha_0 + \alpha_1 \text{Cluster}_i + \alpha_2 \text{Score}_i + \alpha_3 \text{Reputation}_i + \alpha_4 \text{Age}_i + v_{\text{Partner}_i} + \epsilon_i$$

**参数估计结果**：

| 参数 | 定义 | 系数 | Std.Err | z 值 | P 值 | 显著性 |
|-----|-----|------|--------|------|------|-------|
| $\alpha_0$ | 截距 | -1.957 | 0.162 | -12.07 | <0.001 | *** |
| $\alpha_1(\text{Fan Favorite})$ | Fan Favorite 效应 | 0.048 | 0.066 | 0.73 | 0.465 | |
| $\alpha_1(\text{Performance Artist})$ | Performance Artist 效应 | 0.075 | 0.053 | 1.40 | 0.160 | |
| $\alpha_2$ | 技术评分效应 | -0.998 | 0.246 | -4.05 | <0.001 | *** |
| $\alpha_3$ | 初始名气效应 | +3.155 | 0.324 | 9.75 | <0.001 | *** |
| $\alpha_4$ | 年龄效应（标准化） | -0.182 | 0.025 | -7.24 | <0.001 | *** |

**关键发现**：

- 🌟 **名气主导**：初始名气是最强预测因子（系数 +3.16）
- ⚖️ **行业无差异**：行业簇对粉丝投票无显著影响（P > 0.05）
- 🎭 **技术-人气悖论**：技术评分越高，粉丝投票越少（系数 -1.0）
  - 解释：技术高的选手被认为"不需要帮助"，观众更愿意投给 underdog
- 👴 **年龄惩罚加剧**：年龄对人气的负面影响是对技术的 **4 倍**

---

### 4.5 年龄线性检验

#### 4.5.1 线性 vs 二次模型对比

**公式**：

$$\text{Model}_{\text{linear}}: Y = \beta_0 + \beta_1 \text{Age} + \epsilon$$

$$\text{Model}_{\text{quadratic}}: Y = \beta_0 + \beta_1 \text{Age} + \beta_2 \text{Age}^2 + \epsilon$$

**结果对比**：

| 赛道 | 线性 LogLik | 二次 LogLik | Age² 系数 | Age² P 值 | 推荐 |
|-----|------------|------------|----------|----------|------|
| 技术评分 | 1706.99 | 1703.13 | -0.0033 | **0.0892** | 线性 ✓ |
| 粉丝投票 | -4304.39 | -4307.50 | -0.0063 | **0.7029** | 线性 ✓ |

**结论**：两个模型的二次项 P 值均 > 0.05，**线性假设成立**，无需引入二次项。

---

### 4.6 盲投效应分析（Blind Voting Effect）

#### 4.6.1 代理变量构造

**公式**：

$$D_{\text{Blind}} = \begin{cases}
1 & \text{if Season} \in \{28, 29, 30\} \\
0 & \text{otherwise}
\end{cases}$$

**背景**：S28-S30 引入 Live Vote，但西海岸有 3 小时时差，导致部分观众在未看到表演前就投票。

#### 4.6.2 交互效应模型

**公式**：

$$Y_{\text{Fan}} = \alpha_0 + \alpha_1 \text{Score} + \alpha_2 (\text{Score} \times D_{\text{Blind}}) + \alpha_3 \text{Rep} + \alpha_4 (\text{Rep} \times D_{\text{Blind}}) + \sum \alpha_k \text{Cluster}_k + v_{\text{Partner}}$$

**参数估计结果**：

| 参数 | 定义 | 系数 | Std.Err | P 值 | 解读 |
|-----|-----|------|--------|------|------|
| $\alpha_1$ (Score) | 基准：评分对投票的影响 | -0.672 | 0.262 | 0.010** | 基准：分数高投票少 |
| **$\alpha_2$ (Score × D\_Blind)** | 盲投脱钩效应 | **-3.188** | 0.831 | **<0.001*** | 盲投时评分影响大幅削弱 |
| $\alpha_3$ (Reputation) | 基准：声誉对投票的影响 | +2.955 | 0.342 | <0.001*** | 声誉强烈驱动投票 |
| **$\alpha_4$ (Rep × D\_Blind)** | 声誉替代效应 | **+3.063** | 1.192 | **0.010*** | 盲投时声誉影响显著增强 |

**核心发现**：

- ✅ **假设1验证**：盲投时期，当前表现对投票的影响**大幅削弱**（-3.19）
- ✅ **假设2验证**：盲投时期，历史声誉对投票的影响**显著增强**（+3.06）

---

### 4.7 专业舞伴效应（Kingmaker Effect）

#### 4.7.1 技术提升 Top 3

| 舞伴 | 技术加成 | 说明 |
|-----|---------|------|
| Derek Hough | +0.075 | 最佳技术加成 |
| Artem Chigvintsev | +0.055 | 第二名 |
| Valentin Chmerkovskiy | +0.042 | 第三名 |

#### 4.7.2 人气提升 Top 3

| 舞伴 | 人气加成 | 说明 |
|-----|---------|------|
| Derek Hough | +0.123 | 最佳人气加成 |
| Jenna Johnson | +0.070 | 第二名 |
| Valentin Chmerkovskiy | +0.061 | 第三名 |

**结论**：**Derek Hough 是唯一同时登顶技术和人气榜的"双料 Kingmaker"**

---

## 5. Question 4：新赛制设计与验证

### 5.1 问题定义与建模目标

**核心问题**：如何设计一种更公平的赛制，在保证娱乐性的同时保护高技术选手？

**建模目标**：

1. 设计动态门槛百分比模型（DTPM）
2. 通过 Pareto 优化找到最优参数组合
3. 用历史数据进行回测验证

### 5.2 动态门槛百分比模型（DTPM）设计理念

**核心目标**：修复现有赛制的公平性漏洞

**设计原则**：

1. **初期严把关**：前几周评委掌握绝对话语权，确保留下的一定是会跳舞的
2. **差生打折**：表现极差的选手粉丝票自动打折扣，防止"混子"靠粉丝基数获胜

---

### 5.3 机制 A：动态权重（Dynamic Weighting）

#### 5.3.1 公式

$$w(t) = w_{\text{start}} - \frac{t-1}{T-1} \times (w_{\text{start}} - w_{\text{end}})$$

**参数说明**：

| 参数符号 | 物理意义 | 最优值 | 说明 |
|---------|---------|-------|------|
| $w(t)$ | 第 $t$ 周的评委权重 | 动态变化 | 线性递减 |
| $w_{\text{start}}$ | 初始评委权重 | 0.90 | 第 1 周评委掌握 90% 话语权 |
| $w_{\text{end}}$ | 最终评委权重 | 0.60 | 决赛周评委掌握 60% 话语权 |
| $t$ | 当前周次 | 1 到 $T$ | 赛季中的周次编号 |
| $T$ | 赛季总周数 | 因赛季而异 | 整季的周数 |

**物理意义**：

- 第 1 周：评委权重 = 0.90（90% 评委 + 10% 粉丝）
- 决赛周：评委权重 = 0.60（60% 评委 + 40% 粉丝）
- 线性递减，"先严后松"

---

### 5.4 机制 B：表现门槛乘数（Performance Gated Multiplier）

#### 5.4.1 公式

$$\gamma_i = \begin{cases}
1.0 & \text{if JudgeScore}_i \geq 0.8 \times \mu_{\text{judge}} \\
\beta & \text{if JudgeScore}_i < 0.8 \times \mu_{\text{judge}}
\end{cases}$$

**参数说明**：

| 参数符号 | 物理意义 | 最优值 | 说明 |
|---------|---------|-------|------|
| $\gamma_i$ | 选手 $i$ 的表现乘数 | 1.0 或 0.4 | 1.0=正常，0.4=打四折 |
| $\text{JudgeScore}_i$ | 选手 $i$ 当周评委分 | 原始数据 | 多支舞蹈分数之和 |
| $\mu_{\text{judge}}$ | 当周评委平均分 | $\frac{1}{n}\sum \text{JudgeScore}_j$ | 衡量当周整体水平 |
| $\beta$ | 惩罚系数 | 0.40 | 低于门槛时的折扣力度 |

**逻辑解读**：

- 评委分 ≥ 80% 平均分：乘数 = 1.0（正常计算）
- 评委分 < 80% 平均分：乘数 = 0.4（粉丝票打四折）

**设计原理**：允许选手有轻微失误（"Average-ish"），但对于真正"极其糟糕"的表现（如 Bobby Bones 常态），由于触发了 20% 红线，粉丝票会被熔断。

---

### 5.5 DTPM 总分公式

#### 5.5.1 综合得分公式

$$\text{DTPM Score}_i = w(t) \times P_{\text{judge},i} + (1 - w(t)) \times (P_{\text{fan},i} \times \gamma_i)$$

**参数说明**：

| 参数符号 | 物理意义 | 说明 |
|---------|---------|------|
| $\text{DTPM Score}_i$ | 选手 $i$ 的 DTPM 综合得分 | 归一化后用于排名 |
| $w(t)$ | 第 $t$ 周的评委权重 | 动态变化 |
| $P_{\text{judge},i}$ | 选手 $i$ 的评委分占比 | $\frac{\text{JudgeScore}_i}{\sum \text{JudgeScore}}$ |
| $P_{\text{fan},i}$ | 选手 $i$ 的粉丝票占比 | $\frac{\text{FanVotes}_i}{\sum \text{FanVotes}}$ |
| $\gamma_i$ | 选手 $i$ 的表现乘数 | 1.0 或 0.4 |

#### 5.5.2 淘汰规则

$$\text{Eliminated} = \arg\min_i (\text{DTPM Score}_i)$$

得分最低的选手被淘汰。

---

### 5.6 Pareto 优化与参数选择

#### 5.6.1 双目标优化

**目标函数**：

$$\max_{\mathbf{x} \in \mathcal{X}} \quad \{\text{Fairness}(\mathbf{x}), \text{Entertainment}(\mathbf{x})\}$$

其中 $\mathbf{x} = (w_{\text{start}}, w_{\text{end}}, \beta)$ 为参数向量。

**指标定义**：

| 指标 | 计算方式 | 优化方向 |
|-----|---------|---------|
| **Fairness** | Kendall's Tau | 最大化 |
| **Entertainment** | $1 - \text{Upset Rate}$ | 最大化 |

#### 5.6.2 最优参数组合

| 参数 | 最优值 | 物理意义 |
|-----|-------|---------|
| $w_{\text{start}}$ | 0.90 | 第 1 周评委掌握 90% 话语权 |
| $w_{\text{end}}$ | 0.60 | 决赛周评委掌握 60% 话语权 |
| $\beta$ | 0.40 | 表现门槛惩罚系数 |

---

### 5.7 回测验证结果

#### 5.7.1 宏观指标对比

| 评估维度 | 历史现状（Baseline） | 新赛制（DTPM） | 优化效果 |
|---------|---------------------|---------------|---------|
| **公平性** | Kendall's Tau = 0.6248 | **0.6962** | 🟢 **提升 11.4%** |
| **混乱度** | Upset Rate = 56.47% | **24.78%** | 🟢 **降低 56.1%** |

**黄金区间定义**：15%-25% 的逆转率（保留悬念，但不至于"乱选"）

#### 5.7.2 争议案例修正

| 选手（赛季） | 历史战绩 | DTPM 战绩 | 修正效果 |
|------------|---------|----------|---------|
| **Bobby Bones (S27)** | 冠军 | 第 9 周淘汰 | 🚫 剥夺不公冠军，熔断机制生效 |
| **Billy Ray Cyrus (S4)** | 第 5 名 | 第 1 周淘汰 | 📉 提前 7 周淘汰，初期高权重生效 |
| **Bristol Palin (S11)** | 第 3 名 | 第 5 周淘汰 | 📉 提前 5 周淘汰，不再保送 |
| **Jerry Rice (S2)** | 亚军 | 第 7 周淘汰 | 📉 提前 1 周淘汰，未能进入决赛 |

---

## 6. 代码目录与实现说明

### 6.1 Question 1 代码模块

| 文件名 | 行数 | 主要功能 | 核心类/函数 |
|-------|------|---------|------------|
| `monte_carlo_bayesian_dirichlet.py` | ~800 | 贝叶斯-狄利克雷混合模型主程序 | `MonteCarloBayesianDirichlet` |
| `monte_carlo_sim.py` | ~700 | 蒙特卡洛模拟框架 | `MonteCarloSimulation` |
| `data_processing.py` | ~700 | 数据预处理与诊断 | `generate_diagnostic_plots()` |
| `evaluate_models.py` | ~700 | 模型评估与可视化 | `plot_explanation_rate()` |
| `diagnostics.py` | ~1300 | 混沌优化与分布比较 | `run_chaos_optimization()` |

**核心类与方法**：

```python
# monte_carlo_bayesian_dirichlet.py
class MonteCarloBayesianDirichlet:
    def __init__(self, config):  # 初始化配置
    def initialize_priors(self, names):  # 初始化狄利克雷先验
    def update_priors(self, prior, judge_scores):  # 更新先验分布
    def sample(self, n_samples):  # 蒙特卡洛采样
    def fit(self, data):  # 模型拟合
    def predict(self, week):  # 预测投票份额
```

---

### 6.2 Question 2 代码模块

| 文件名 | 行数 | 主要功能 | 核心类/函数 |
|-------|------|---------|------------|
| `rule_comparison.py` | ~1200 | 规则系统对比主程序 | `RuleComparison` |
| `system_diagnostics.py` | ~750 | 系统压力测试 | `run_stress_test()` |
| `analyze_judges_save.py` | ~250 | 评委拯救机制分析 | `analyze_save_probability()` |

**核心类与方法**：

```python
# rule_comparison.py
class RuleComparison:
    def __init__(self, data):  # 初始化数据
    def apply_rank_system(self):  # 应用排名制
    def apply_percent_system(self):  # 应用百分比制
    def apply_rank_with_save(self):  # 应用排名+拯救制
    def calculate_metrics(self):  # 计算公平性指标
    def analyze_celebrity_case(self, name):  # 名人案例分析
```

---

### 6.3 Question 3 代码模块

| 文件名 | 行数 | 主要功能 | 核心类/函数 |
|-------|------|---------|------------|
| `factor_analysis_v2.py` | ~1800 | 因素分析主程序 | `FactorAnalysis` |
| `data_processing.py` | ~700 | 数据预处理 | `process_factor_data()` |

**核心类与方法**：

```python
# factor_analysis_v2.py
class FactorAnalysis:
    def __init__(self, config):  # 初始化配置
    def perform_clustering(self, k=3):  # K-Means 聚类
    def fit_model_a(self):  # 技术轨道模型
    def fit_model_b(self):  # 人气轨道模型
    def analyze_blind_voting(self):  # 盲投效应分析
    def analyze_partner_effects(self):  # 舞伴效应分析
    def test_age_linearity(self):  # 年龄线性检验
```

---

### 6.4 Question 4 代码模块

| 文件名 | 行数 | 主要功能 | 核心类/函数 |
|-------|------|---------|------------|
| `system_design.py` | ~1000 | DTPM 系统设计 | `DTPMOptimizer` |
| `case_study_analysis.py` | ~350 | 争议案例回测 | `run_case_studies()` |
| `evaluate_baseline.py` | ~150 | 基准评估 | `evaluate_current_system()` |
| `sensitivity_analysis_enhanced.py` | ~1000 | 灵敏度分析 | `q4_dtpm_backtesting()` |

**核心类与方法**：

```python
# system_design.py
class DTPMOptimizer:
    def __init__(self, config):  # 初始化配置
    def set_weights(self, w_start, w_end):  # 设置动态权重
    def set_threshold_multiplier(self, beta):  # 设置门槛乘数
    def calculate_score(self, contestant):  # 计算 DTPM 得分
    def pareto_optimize(self):  # Pareto 优化
    def backtest(self, historical_data):  # 历史回测
```

---

### 6.5 公共工具模块

| 文件名 | 行数 | 主要功能 |
|-------|------|---------|
| `viz_config.py` | ~300 | 可视化配色与样式配置 |
| `data_processing.py` | ~700 | 共享数据预处理函数 |

**可视化配置示例**：

```python
# viz_config.py
MORANDI_COLORS = ['#7DA9C7', '#A8D5BA', '#E8C5A0', '#D4A8C7', '#B5C9D0']
MORANDI_ACCENT = ['#5B8FB9', '#3A7D44', '#D4A373', '#9B5DE5']
```

---

## 7. 附录：参数符号表与指标定义

### 7.1 希腊字母参数汇总

| 符号 | 读音 | 物理意义 | 典型取值 |
|-----|------|---------|---------|
| $\alpha$ | alpha | 狄利克雷先验参数 / 行业评分 | 0.1-5.0 |
| $\beta$ | beta | 惩罚系数 / 回归系数 | 0.4 |
| $\gamma$ | gamma | 表现门槛乘数 | 0.4 或 1.0 |
| $\delta$ | delta | 差分算子 | — |
| $\epsilon$ | epsilon | 随机误差项 | $\epsilon \sim N(0, \sigma^2)$ |
| $\eta$ | eta | 学习率 | 0.4 |
| $\theta$ | theta | 一般参数向量 | — |
| $\lambda$ | lambda | 混沌权重 | 0.024 |
| $\mu$ | mu | 均值 | 计算得出 |
| $\rho$ | rho | Spearman 相关系数 | [-1, 1] |
| $\sigma$ | sigma | 标准差 | 计算得出 |
| $\tau$ | tau | Kendall's Tau | [-1, 1] |

### 6.2 关键指标定义

#### A.1 解释率（Explanation Rate）

$$\text{Explanation Rate} = \frac{\text{成功复现的淘汰周数}}{\text{总消除周数}} \times 100\%$$

**含义**：模型能够解释多少历史淘汰结果。数值越高，模型解释能力越强。

---

#### A.2 置信度（Certainty）

$$\text{Certainty} = \frac{n_{\text{valid\_sims}}}{N_{\text{total}}}$$

**含义**：在 $N_{\text{total}}$ 次模拟中，有多少次产生了有效预测。数值越高，模型越确定。

---

#### A.3 香农熵（Shannon Entropy）

$$H(X) = -\sum_{i=1}^{n} p_i \ln(p_i)$$

**含义**：衡量概率分布的不确定性。单位：nats（自然对数单位）。

---

#### A.4 信息比率（Information Ratio）

$$\text{IR} = \frac{\text{Explanation Rate}}{1 - \text{Certainty}}$$

**含义**：在解释能力和自信度之间寻找最优平衡。分母是"不确定度"。

---

#### A.5 Kendall's Tau

$$\tau = \frac{C - D}{\frac{1}{2}n(n-1)}$$

其中：
- $C$：和谐对数量（两变量顺序一致的对数）
- $D$：不和谐对数量（两变量顺序不一致的对数）
- $n$：样本数量

**含义**：衡量两个排名之间的一致性。1 = 完全一致，-1 = 完全相反。

---

#### A.6 轮廓系数（Silhouette Score）

$$s = \frac{b - a}{\max(a, b)}$$

其中：
- $a$：簇内平均距离
- $b$：最近簇的平均距离

**含义**：衡量聚类质量。范围：[-1, 1]，越接近 1 越好。

---

#### A.7 调整兰德指数（Adjusted Rand Index, ARI）

$$\text{ARI} = \frac{\text{RI} - E[\text{RI}]}{\max(\text{RI}) - E[\text{RI}]}$$

**含义**：衡量两个聚类结果的相似度，经过随机聚类校正。范围：[-1, 1]，1 = 完全一致。

---

**文档维护**：
- 最后更新：2026-02-03
- 相关代码：`src/monte_carlo_bayesian_dirichlet.py`、`src/rule_comparison.py`、`src/factor_analysis_v2.py`、`src/case_study_analysis.py`
- 结果目录：`results/plots/`、`results/question2/`、`results/question3/`、`results/system_design/`
