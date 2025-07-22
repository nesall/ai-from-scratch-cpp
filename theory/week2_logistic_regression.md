# Week 2: Logistic Regression (from Scratch in C++)

## 🧠 Goal
Classify input data into two classes (`0` or `1`) using a linear model + sigmoid function.

---

## 🔢 Model

**Linear part:**

\[
z = wx + b
\]

**Sigmoid activation:**

\[
\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}
\]

Where:
- \( \hat{y} \in (0, 1) \) is interpreted as the probability of class `1`.

---

## 📉 Loss Function

**Binary Cross-Entropy Loss:**

\[
\text{Loss} = -\frac{1}{n} \sum \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
\]

---

## 🧮 Training: Gradient Descent

**Gradients:**

\[
\begin{align*}
\frac{\partial L}{\partial w} &= \frac{1}{n} \sum ( \hat{y}_i - y_i ) x_i \\
\frac{\partial L}{\partial b} &= \frac{1}{n} \sum ( \hat{y}_i - y_i )
\end{align*}
\]

**Update:**

\[
\begin{align*}
w &:= w - \alpha \cdot \frac{\partial L}{\partial w} \\
b &:= b - \alpha \cdot \frac{\partial L}{\partial b}
\end{align*}
\]

---

## ✅ Implementation Plan

- Use `float` for weights and biases
- Create `sigmoid()` helper
- Add methods:
  - `fit()`
  - `predict_proba()`
  - `predict()`