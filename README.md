# Statistical Arbitrage: Bitcoin vs Nvidia

This project implements a statistical arbitrage strategy comparing the volatility of Bitcoin and Nvidia using rolling statistics and a polynomial regression model.

---

## Mathematical Formulas

### 1. **Daily Return**

Calculates the daily rate of return for asset prices:
$
r*t = \frac{P_t}{P*{t-1}} - 1
$

---

### 2. **Rolling Volatility**

Measures standard deviation of returns over a rolling window:
$
\sigma = \sqrt{\frac{1}{n} \sum\_{i=1}^{n}(x_i - \bar{x})^2}
$

---

### 3. **Second-Degree Polynomial Regression**

Fits the relationship:
$
Y = \beta_0 + \beta_1 X + \beta_2 X^2
$
Using normal equations:
$
\boldsymbol{\beta} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}
$

---

### 4. **Residual Sum of Squares (RSS)**

Measures error of the regression:
$
\text{RSS} = \sum\_{i=1}^{n} (y_i - \hat{y}\_i)^2
$

---

### 5. **Residual Variance Estimate**

Estimates noise in the regression:
$
\text{Var}(\epsilon) = \frac{\text{RSS}}{n - p}
$
Where \( p = 3 \) (including intercept and two polynomial terms).

---

### 6. **Standard Error of Coefficients**

$
\text{SE}(\beta*i) = \sqrt{ \text{Var}(\epsilon) \cdot \left[(\mathbf{X}^\top \mathbf{X})^{-1}\right]*{ii} }
$

---

### 7. **t-Statistic**

Used to assess significance of each coefficient:
$
t_i = \frac{\beta_i}{\text{SE}(\beta_i)}
$

---

### 8. **p-Value Estimation**

Based on normal distribution approximation:
$
p_i = \begin{cases}
\Phi(t_i) & \text{if } t_i < 0.5 \\
1 - \Phi(t_i) & \text{otherwise}
\end{cases}
$

---

### 9. **Z-Score of Regression Spread**

Standardizes the current spread from its historical mean:
$
Z = \frac{s\_{t} - \mu_s}{\sigma_s / \sqrt{n}}
$

---

### 10. **Trade Volume Calculation**

Using 90% of capital:
$
\text{Volume}\_{\text{asset}} = \frac{0.9 \cdot \text{Balance}}{\text{Price}}
$

---

### 11. **Balance Update (Including Fees & Interest)**

Adjusts for fees and holding costs:
$
\text{Adjusted Value} = \text{Entry Price} \cdot (1 - \text{Interest}) \cdot (1 - \text{Fee}) - \text{Exit Price} \cdot (1 + \text{Fee})
$

---

## Notes

- Transaction Fee: **0.2%**
- Interest Rate on Held Capital: **0.05%**
- Signals triggered by Z-score threshold of **±30**
- Strategy assumes zero slippage
