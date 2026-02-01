# Task 3: Factor Analysis Key Findings

## 1. The 'Partner Effect' (Kingmaker Analysis)
### Top Technical Instructors (Judge Bonus):
- **Derek Hough**: +0.103 score boost
- **Artem Chigvintsev**: +0.070 score boost
- **Valentin Chmerkovskiy**: +0.054 score boost

### Top Fan Favorites (Vote Magnets):
- **Derek Hough**: +0.406 log-share boost
- **Valentin Chmerkovskiy**: +0.182 log-share boost
- **Jenna Johnson**: +0.176 log-share boost

## 2. Demographic Biases
### Age Effect (Linear View):
- **Skill**: -0.0581 (per SD of age)
- **Popularity**: -0.2683 (per SD of age)

### Age Non-linearity Analysis:
#### Skill Linearity Test:
- Linear AIC: nan | Quadratic AIC: nan
- Quadratic term p-value: 0.0752
- **Conclusion**: Linear effect is more appropriate.
#### Popularity Linearity Test:
- Linear AIC: nan | Quadratic AIC: nan
- Quadratic term p-value: 0.6872
- **Conclusion**: Linear effect is more appropriate.

### Industry Effect (vs Actor):
- **Skill (C(industry_group)[T.Athlete])**: -0.029
- **Skill (C(industry_group)[T.Model])**: -0.093
- **Skill (C(industry_group)[T.Other])**: -0.045
- **Pop (C(industry_group)[T.Athlete])**: -0.151
- **Pop (C(industry_group)[T.Model])**: -0.390
- **Pop (C(industry_group)[T.Other])**: -0.164
