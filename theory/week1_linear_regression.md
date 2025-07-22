# Week 1: Linear Regression (from Scratch in C++)

## 🧠 Goal
Model a relationship between input variable `x` and a continuous output `y` using a linear function.

---

## 🔢 Model

**Prediction formula:**

\[
\hat{y} = wx + b
\]

Where:
- \( \hat{y} \) is the predicted output
- \( w \) is the weight
- \( b \) is the bias/intercept

---

## 🎯 Loss Function

**Mean Squared Error (MSE):**

\[
\text{Loss} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2
\]

---

## 🧮 Training: Gradient Descent

**Weight update rules:**

\[
\begin{align*}
\frac{\partial L}{\partial w} &= \frac{2}{n} \sum (\hat{y}_i - y_i) x_i \\
\frac{\partial L}{\partial b} &= \frac{2}{n} \sum (\hat{y}_i - y_i)
\end{align*}
\]

**Parameter update:**

\[
\begin{align*}
w &:= w - \alpha \cdot \frac{\partial L}{\partial w} \\
b &:= b - \alpha \cdot \frac{\partial L}{\partial b}
\end{align*}
\]

Where \( \alpha \) is the learning rate.

---

## ✅ Implementation Summary

- Fit using batch gradient descent
- Predict single value or a vector of values
- Validate with MSE
- Optional early stopping if MSE stops improving