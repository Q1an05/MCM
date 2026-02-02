# Task 3 Advanced Analysis: Key Findings

## 1. Industry Clustering (K-Means)
Industries were classified into clusters based on 3 dimensions:
- **Physicality**: Physical training requirements
- **Performance**: Stage presence and emotional expression
- **Fanbase**: Built-in fan loyalty

### Cluster Distribution:
- **Performance Artist**: 1547 observations (55.7%)
- **Athletic Elite**: 683 observations (24.6%)
- **Fan Favorite**: 547 observations (19.7%)

## 2. Growth Trajectory Analysis
### Average Growth Slope by Cluster:
- **Athletic Elite**: 0.0270 (n=78)
- **Fan Favorite**: 0.0298 (n=66)
- **Performance Artist**: 0.0262 (n=182)

## 3. Top Professional Partners (Kingmaker Effect)
### Top Technical Instructors:
- **Derek Hough**: +0.075 skill boost
- **Artem Chigvintsev**: +0.055 skill boost
- **Valentin Chmerkovskiy**: +0.042 skill boost

### Top Fan Magnets:
- **Derek Hough**: +0.123 popularity boost
- **Jenna Johnson**: +0.070 popularity boost
- **Valentin Chmerkovskiy**: +0.061 popularity boost

## 4. Model Coefficients
### Technical Model (Judge Score):
- C(industry_cluster)[T.Fan Favorite]: -0.0482 ***
- C(industry_cluster)[T.Performance Artist]: 0.0068 
- week: 0.0305 ***
- C(industry_cluster)[T.Fan Favorite]:week: 0.0042 *
- C(industry_cluster)[T.Performance Artist]:week: 0.0003 
- age_std: -0.0435 ***

### Popularity Model (Fan Share):
- C(industry_cluster)[T.Fan Favorite]: 0.0485 
- C(industry_cluster)[T.Performance Artist]: 0.0748 
- normalized_score: -0.9978 ***
- reputation: 3.1548 ***
- age_std: -0.1816 ***

## 5. Blind Voting Effect Analysis
**Definition**: Seasons 28-30 (Live Vote era with time zone delay) as 'Blind Voting' proxy.

### Score-Vote Correlation by Era:
- **Non-Blind Era**: r = 0.2290 (n=2465)
- **Blind Era (S28-S30)**: r = 0.0132 (n=312)

**Result**: Blind Era correlation is 94.2% weaker

### Interaction Test (Score × BlindEra):
- Coefficient: -3.1879
- P-value: 0.0001 ✓

### Interaction Test (Reputation × BlindEra):
- Coefficient: 3.0628
- P-value: 0.0102 ✓
