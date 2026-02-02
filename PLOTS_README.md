# 图片-含义对照表

> **文档说明**：本表格整理了 `results/plots` 文件夹中所有图片的含义，方便团队成员快速查找和理解每张图的内容。
>
> **相关文件**：
>
> - 代码位置：`src/` 文件夹
> - 题目信息：`2026_MCM_Problem_C.pdf`
> - 各题总结：`q1/q2/q3/q4_summary.md`

---

## Section 0 - Data Preprocessing & Exploratory Analysis (数据预处理与探索性分析)

| 图片文件名                     | 代码位置                 | 代码函数                      | 图片含义                                                                                                                                            |
| ------------------------------ | ------------------------ | ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `data_diagnostics_summary.png` | `src/data_processing.py` | `generate_diagnostic_plots()` | **数据诊断总览**：整合三个子图：(1) 评委平均总分热力图（赛季 x 周）；(2) 每季参赛人数趋势；(3) 每季赛程周数。用于直观展示原始数据的分布与缺失情况。 |
| `top_15_industries.png`        | `src/data_processing.py` | `generate_diagnostic_plots()` | **Top 15 行业分布**：展示选手中最常见的 15 个行业类别及其人数，使用淡蓝到深蓝的渐变配色及数据标注。                                                 |

---

## Question 1 - 模型评估与诊断

### 1.1 核心评估图表

| 图片文件名                        | 代码位置                 | 代码函数                        | 图片含义                                                                                                                                                                      |
| --------------------------------- | ------------------------ | ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `explanation_rate_comparison.png` | `src/evaluate_models.py` | `plot_explanation_rate()`       | **模型解释率对比**：比较 Basic 和 Bayesian 模型在不同规则系统（Percent/Rank/Rank_With_Save）下的解释率（有效消除周数/总消除周数）。解释率越高，说明模型越能准确预测淘汰结果。 |
| `certainty_distribution.png`      | `src/evaluate_models.py` | `plot_certainty_distribution()` | **置信度分布**：展示模型在已解释周数上的置信度分布。置信度 = 有效模拟数/总模拟数。数值越高说明模型对该周预测越有信心。用于比较两种模型的不确定性处理能力。                    |
| `stability_distribution.png`      | `src/evaluate_models.py` | `plot_stability_distribution()` | **稳定性/精确度分布**：比较模型估计的粉丝份额标准差。标准差越低，说明模型估计越稳定精确。这是评估模型质量的关键指标。                                                         |
| `entropy_distribution.png`        | `src/evaluate_models.py` | `plot_entropy_distribution()`   | **不确定性特征分布**：使用香农熵（Shannon Entropy）展示模型的不确定性特征。熵值越高表示模型对该周参赛者的粉丝分布越不确定。主要用于 Bayesian 模型。                           |

### 1.2 熵热图分析

| 图片文件名                | 代码位置                 | 代码函数                 | 图片含义                                                                                                                                                                     |
| ------------------------- | ------------------------ | ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `entropy_heatmap_S27.png` | `src/evaluate_models.py` | `plot_entropy_heatmap()` | **Season 27 熵热图**：Bobby Bones 时代的熵热图。X轴=周数，Y轴=参赛者，颜色深浅表示熵值（深色=高不确定性/潜在黑马，浅色=低不确定性/被淘汰边缘）。用于识别每周的不确定性分布。 |
| `entropy_heatmap_S28.png` | `src/evaluate_models.py` | `plot_entropy_heatmap()` | **Season 28 熵热图**：Sean Spicer 时代的熵热图。展示该赛季每周各参赛者的不确定性水平，帮助分析淘汰模式。                                                                     |
| `entropy_heatmap_S32.png` | `src/evaluate_models.py` | `plot_entropy_heatmap()` | **Season 32 熵热图**：Xochitl Gomez / Jason Mraz 时代的熵热图。近期高技能赛季的熵分析，展示高水平选手间的不确定性差异。                                                      |

### 1.3 案例轨迹分析

| 图片文件名                       | 代码位置                 | 代码函数                | 图片含义                                                                                                                                             |
| -------------------------------- | ------------------------ | ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `trajectory_Bobby_Bones_S27.png` | `src/evaluate_models.py` | `plot_fan_trajectory()` | **Bobby Bones S27 粉丝份额轨迹**：比较 Basic 和 Bayesian 模型对 Bobby Bones（Season 27）在每周的粉丝份额估计。展示估计值和置信区间随时间的变化趋势。 |
| `trajectory_Sean_Spicer_S28.png` | `src/evaluate_models.py` | `plot_fan_trajectory()` | **Sean Spicer S28 粉丝份额轨迹**：比较两种模型对 Sean Spicer（Season 28）的粉丝份额估计轨迹。分析其在争议赛季中的表现。                              |

### 1.4 案例研究双轴图

| 图片文件名                               | 代码位置                 | 代码函数                         | 图片含义                                                                                                                                             |
| ---------------------------------------- | ------------------------ | -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `case_study_entropy_Bobby_Bones_S27.png` | `src/evaluate_models.py` | `plot_dual_axis_entropy_share()` | **Bobby Bones 案例研究双轴图**：双轴图表，左轴显示粉丝份额，右轴显示熵值。用于分析"投票规模"与"不确定性"之间的关系。揭示模型对高人气选手的预测特征。 |
| `case_study_entropy_Sean_Spicer_S28.png` | `src/evaluate_models.py` | `plot_dual_axis_entropy_share()` | **Sean Spicer 案例研究双轴图**：类似的双轴分析，展示 Sean Spicer 在争议赛季中的粉丝份额与不确定性关系。帮助理解模型对低 judge 评分选手的处理方式。   |

### 1.5 混沌优化分析

| 图片文件名                        | 代码位置             | 代码函数     | 图片含义                                                                                                    |
| --------------------------------- | -------------------- | ------------ | ----------------------------------------------------------------------------------------------------------- |
| `certainty_vs_lambda.png`         | `src/diagnostics.py` | 混沌优化模块 | **置信度 vs Lambda 曲线**：展示在混沌权重 lambda 变化时，模型置信度的变化趋势。用于找到最优的混沌参数设置。 |
| `distribution_comparison.png`     | `src/diagnostics.py` | 混沌优化模块 | **分布对比图**：比较不同分布类型（Uniform/Pareto/Exponential）下的模型表现。用于选择最佳的混沌分布模型。    |
| `explanation_rate_vs_lambda.png`  | `src/diagnostics.py` | 混沌优化模块 | **解释率 vs Lambda 曲线**：展示混沌权重 lambda 对模型解释率的影响。用于优化模型参数以提高解释率。           |
| `information_ratio_vs_lambda.png` | `src/diagnostics.py` | 混沌优化模块 | **信息比率 vs Lambda 曲线**：分析混沌参数对信息比率的影响，用于评估模型的信息效率。                         |
| `skill_bias_scatter.png`          | `src/diagnostics.py` | 混沌优化模块 | **技能偏差散点图**：展示模型在不同技能水平选手上的偏差分布。用于分析模型是否存在系统性偏差。                |

---

## Question 2 - 规则系统对比分析

### 2.1 生存分析与淘汰规则对比

| 图片文件名                            | 代码位置                 | 代码函数       | 图片含义                                                                                                                                     |
| ------------------------------------- | ------------------------ | -------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `q2_bobby_bones_survival.png`         | `src/rule_comparison.py` | 生存分析模块   | **Bobby Bones 生存分析**：使用 Kaplan-Meier 方法分析 Bobby Bones 在不同规则系统下的生存概率。展示其在 Rank + Judges' Save 系统下的淘汰风险。 |
| `q2_celebrity_combined.png`           | `src/rule_comparison.py` | 多选手综合分析 | **名人综合对比图**：综合展示多个争议名人在不同规则系统下的淘汰风险对比。用于比较规则系统对不同类型选手的影响。                               |
| `q2_celebrity_bobby_bones_s27.png`    | `src/rule_comparison.py` | 单选手详细分析 | **Bobby Bones S27 详细分析**：针对 Bobby Bones 在 Season 27 的详细淘汰分析，包括历史表现和模拟预测。                                         |
| `q2_celebrity_jerry_rice_s2.png`      | `src/rule_comparison.py` | 单选手详细分析 | **Jerry Rice S2 详细分析**：针对 Jerry Rice 在 Season 2 的详细淘汰分析。                                                                     |
| `q2_celebrity_bristol_palin_s11.png`  | `src/rule_comparison.py` | 单选手详细分析 | **Bristol Palin S11 详细分析**：针对 Bristol Palin 在 Season 11 的详细淘汰分析。                                                             |
| `q2_celebrity_billy_ray_cyrus_s4.png` | `src/rule_comparison.py` | 单选手详细分析 | **Billy Ray Cyrus S4 详细分析**：针对 Billy Ray Cyrus 在 Season 4 的详细淘汰分析。                                                           |

### 2.2 规则系统评估指标

| 图片文件名                   | 代码位置                 | 代码函数                     | 图片含义                                                                                                  |
| ---------------------------- | ------------------------ | ---------------------------- | --------------------------------------------------------------------------------------------------------- |
| `q2_fan_bias_comparison.png` | `src/rule_comparison.py` | `plot_fan_bias_comparison()` | **粉丝偏差对比**：比较不同规则系统下粉丝投票的影响力。分析规则系统是否更偏向粉丝投票或评委打分。          |
| `q2_merit_metrics.png`       | `src/rule_comparison.py` | `plot_merit_metrics()`       | **能力指标评估**：使用 merit-based 指标评估规则系统的公平性。展示系统是否能够有效区分高能力和低能力选手。 |

### 2.3 系统诊断与压力测试

| 图片文件名                      | 代码位置                    | 代码函数                   | 图片含义                                                                                                                 |
| ------------------------------- | --------------------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `q2_system_stress_test.png`     | `src/system_diagnostics.py` | `run_stress_test()`        | **系统压力测试**：通过注入 Pareto 混沌分布来测试规则系统的稳健性。展示系统在极端条件下的表现变化。                       |
| `q2_real_decision_boundary.png` | `src/system_diagnostics.py` | `plot_decision_boundary()` | **决策边界图**：在（评委分数，粉丝份额）空间中的生存边界图。展示选手在不同区域被淘汰的风险。用于理解规则系统的决策逻辑。 |

---

## Question 3 - 因素分析与行业聚类

### 3.1 行业聚类分析

| 图片文件名                          | 代码位置                    | 代码函数                      | 图片含义                                                                                                          |
| ----------------------------------- | --------------------------- | ----------------------------- | ----------------------------------------------------------------------------------------------------------------- |
|                                     |                             |                               |                                                                                                                   |
| `q3_industry_clustering_3d.png`     | `src/factor_analysis_v2.py` | `perform_kmeans_clustering()` | **行业聚类 3D 可视化**：在原始 3D 特征空间（Physicality/Performance/Fanbase）中的聚类分布。提供更直观的空间理解。 |
| `q3_cluster_attribute_profiles.png` | `src/factor_analysis_v2.py` | `plot_cluster_profiles()`     | **聚类属性轮廓**：雷达图或柱状图展示每个聚类在三个维度上的平均得分。帮助理解每个聚类的特征。                      |

### 3.2 K值选择与聚类评估

| 图片文件名                   | 代码位置                    | 代码函数             | 图片含义                                                                                             |
| ---------------------------- | --------------------------- | -------------------- | ---------------------------------------------------------------------------------------------------- |
| `q3_optimal_k_selection.png` | `src/factor_analysis_v2.py` | `select_optimal_k()` | **最优 K 值选择图**：使用轮廓系数（Silhouette Score）确定最佳聚类数 K。展示不同 K 值对应的聚类质量。 |

### 3.3 增长轨迹与年龄效应

| 图片文件名                   | 代码位置                    | 代码函数                     | 图片含义                                                                                     |
| ---------------------------- | --------------------------- | ---------------------------- | -------------------------------------------------------------------------------------------- |
| `q3_growth_trajectories.png` | `src/factor_analysis_v2.py` | `plot_growth_trajectories()` | **增长轨迹图**：展示参赛者得分随时间的变化趋势。按聚类分组，比较不同行业类型选手的成长模式。 |
| `q3_age_linearity_test.png`  | `src/factor_analysis_v2.py` | `test_age_linearity()`       | **年龄线性检验**：检验年龄与表现之间是否存在线性关系。通过回归分析展示年龄效应。             |

### 3.4 盲投票与合作伙伴效应

| 图片文件名                   | 代码位置                    | 代码函数                    | 图片含义                                                                                   |
| ---------------------------- | --------------------------- | --------------------------- | ------------------------------------------------------------------------------------------ |
| `q3_blind_voting_effect.png` | `src/factor_analysis_v2.py` | `analyze_blind_voting()`    | **盲投票效应分析**：分析如果采用盲投票（隐藏身份），对不同行业选手的影响差异。             |
| `q3_forest_plot.png`         | `src/factor_analysis_v2.py` | `generate_forest_plot()`    | **森林图**：展示多个因素对淘汰风险的效应大小和置信区间。用于直观比较不同因素的相对重要性。 |
| `q3_partner_effects_v2.png`  | `src/factor_analysis_v2.py` | `analyze_partner_effects()` | **合作伙伴效应**：分析专业舞伴对选手表现的影响。展示有/无专业舞伴的选手表现差异。          |

---

## Question 4 - 系统设计与案例研究

### 4.1 案例对比分析

| 图片文件名                  | 代码位置                     | 代码函数                       | 图片含义                                                                                                                                                                  |
| --------------------------- | ---------------------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `case_study_comparison.png` | `src/case_study_analysis.py` | `plot_case_study_comparison()` | **争议案例对比**：对比分析四个争议案例（Jerry Rice, Billy Ray Cyrus, Bristol Palin, Bobby Bones）在新系统（DTPM）下的淘汰周数与实际结果。展示新系统对历史争议案例的影响。 |

### 4.2 系统优化分析

| 图片文件名                       | 代码位置                                             | 代码函数     | 图片含义                                                                                                        |
| -------------------------------- | ---------------------------------------------------- | ------------ | --------------------------------------------------------------------------------------------------------------- |
| `system_optimization_pareto.png` | `src/case_study_analysis.py` 或 `src/diagnostics.py` | 系统优化模块 | **Pareto 最优边界**：展示系统参数优化的 Pareto 前沿面。在多个目标（解释率、稳定性、公平性）之间找到最优平衡点。 |

---

## Section 5 - 灵敏度分析与稳健性检验 (Sensitivity Analysis & Robustness)

> **新增说明**：本节包含基于真实数据的增强灵敏度分析，使用 Bootstrap 方法估计置信区间，统计显著性检验，以及回测验证。详见 `src/sensitivity_analysis_enhanced.py`。

### 5.1 Question 1 Bootstrap 置信区间分析

| 图片文件名                           | 代码位置                      | 代码函数                      | 图片含义                                                                                                                                                                          |
| ------------------------------------ | ----------------------------- | ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `q1_bootstrap_confidence_intervals.png` | `src/sensitivity_analysis_enhanced.py` | `q1_model_bootstrap_analysis()` | **Bootstrap 置信区间**：对 Bayesian 模型在三种规则系统下的置信度、解释率进行 1000 次 Bootstrap 重采样，计算 95% 置信区间。用于评估模型估计的稳健性和不确定性。                          |
| `q1_stability_by_season.png`         | `src/sensitivity_analysis_enhanced.py` | `q1_model_bootstrap_analysis()` | **按赛季稳定性分析**：展示不同赛季下模型置信度和有效模拟数的稳定性。使用 Wilson Score Interval 估计比例置信区间。用于识别哪些赛季的预测更可靠。                                      |

### 5.2 Question 2 统计显著性检验

| 图片文件名                           | 代码位置                      | 代码函数                         | 图片含义                                                                                                                                                                          |
| ------------------------------------ | ----------------------------- | -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `q2_statistical_tests.png`           | `src/sensitivity_analysis_enhanced.py` | `q2_rule_comparison_statistical_tests()` | **统计显著性检验**：使用卡方检验（Chi-square test）评估规则系统与淘汰结果之间的独立性。展示转换率和 Wilson 置信区间。用于验证规则系统对结果的统计显著性影响。                      |
| `q2_robustness_by_season.png`        | `src/sensitivity_analysis_enhanced.py` | `q2_rule_comparison_statistical_tests()` | **按赛季稳健性分析**：检验不同赛季下规则系统比较结果的稳健性。计算各赛季的转换率及其置信区间，展示统计显著性的季节变化。                                                             |

### 5.3 Question 3 聚类稳健性分析 (Bootstrap)

| 图片文件名                           | 代码位置                      | 代码函数                      | 图片含义                                                                                                                                                                          |
| ------------------------------------ | ----------------------------- | ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `q3_clustering_bootstrap.png`        | `src/sensitivity_analysis_enhanced.py` | `q3_clustering_bootstrap_analysis()` | **聚类 Bootstrap 稳定性**：对行业聚类进行 100 次 Bootstrap 重采样，计算调整兰德指数（ARI）评估聚类稳健性。展示不同聚类数 K 下的稳定性热力图。用于验证聚类结果的可靠性。             |
| `q3_assumption_validation.png`       | `src/sensitivity_analysis_enhanced.py` | `q3_clustering_bootstrap_analysis()` | **聚类假设验证**：验证行业聚类的基本假设（Physicality/Performance/Fanbase 维度）。使用 Bootstrap 置信区间检验各聚类中心是否显著不同。验证聚类分析的合理性。                        |

### 5.4 Question 4 DTPM 回测验证与参数敏感性

| 图片文件名                           | 代码位置                      | 代码函数                      | 图片含义                                                                                                                                                                          |
| ------------------------------------ | ----------------------------- | ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `q4_dtpm_backtesting.png`            | `src/sensitivity_analysis_enhanced.py` | `q4_dtpm_backtesting()`       | **DTPM 回测分析**：对 420 组 DTPM 参数组合（w_start, w_end, beta）进行历史回测。展示参数敏感性热力图，分析各参数对 upset_rate 和公平性指标的影响。验证参数设置的稳健性。             |
| `q4_pareto_frontier.png`             | `src/sensitivity_analysis_enhanced.py` | `q4_pareto_frontier_analysis()` | **Pareto 前沿分析**：在公平性（kendall_tau）与娱乐性（1-upset_rate）之间寻找 Pareto 最优边界。展示参数空间的权衡关系，为系统设计提供决策依据。                                      |

---

## 附录

### A. 图表代码索引

| Python 文件                  | 主要功能                                |
| ---------------------------- | --------------------------------------- |
| `src/evaluate_models.py`     | Question 1 模型评估、熵热图、轨迹分析   |
| `src/diagnostics.py`         | 混沌优化分析、分布比较                  |
| `src/rule_comparison.py`     | Question 2 规则系统对比、生存分析       |
| `src/system_diagnostics.py`  | Question 2 系统压力测试、决策边界       |
| `src/factor_analysis_v2.py`  | Question 3 行业聚类、增长轨迹、因素分析 |
| `src/case_study_analysis.py` | Question 4 争议案例研究、DTPM 系统分析  |

### B. 关键指标说明

| 指标名称                        | 说明                   | 计算方式                    |
| ------------------------------- | ---------------------- | --------------------------- |
| **解释率 (Explanation Rate)**   | 模型成功预测淘汰的比例 | 有效消除周数 / 总消除周数   |
| **置信度 (Certainty)**          | 模型对预测的确信程度   | 有效模拟数 / 总模拟数       |
| **稳定性 (Stability)**          | 模型估计的波动程度     | 粉丝份额标准差（越低越好）  |
| **熵 (Entropy)**                | 不确定性度量           | 香农熵（nats）              |
| **轮廓系数 (Silhouette Score)** | 聚类质量评估           | 范围 [-1, 1]，越接近 1 越好 |
