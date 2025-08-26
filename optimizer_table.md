# Optimizer Recommendations for Models

| Week | Model | Training Method | Recommended Optimizers | Notes |
|------|-------|----------------|----------------------|-------|
| 1 | Linear Regression | Gradient descent on MSE | **SGD** | Sufficient for convex problem; Adam unnecessary |
| 2 | Logistic Regression | Gradient descent on cross-entropy | **SGD**, Momentum | Momentum helps with noisy data |
| 3 | k-Nearest Neighbors | No training (instance-based) | N/A | - |
| 4 | Naive Bayes | Probability estimation | N/A | - |
| 5 | Decision Tree | Greedy splitting | N/A | - |
| 6 | Random Forest | Ensemble of trees | N/A | - |
| 7 | Support Vector Machine (SVM) | Gradient descent on hinge loss | **SGD**, Momentum | Momentum for faster convergence |
| 8 | Perceptron | Gradient descent on classification error | **SGD**, Momentum | Momentum reduces oscillation |
| 9 | Multilayer Perceptron (MLP) | Backpropagation with nonlinear activations | **Adam**, RMSProp | Plain SGD usually too slow |
| 10 | MNIST with MLP | Large dataset, deeper MLP | **Adam**, SGD + Momentum | Adam as default; SGD + Momentum for comparison |
| 11 | Convolutional Neural Network (CNN) | Backpropagation through convolution layers | **Adam**, RMSProp | Adam most common |
| 12 | Recurrent Neural Network (RNN) | Backpropagation Through Time (BPTT) | **Adam**, RMSProp | Adam handles vanishing gradients best; SGD struggles |

## 🔑 Quick Reference

| Model Category | Recommended Optimizer | Reasoning |
|----------------|----------------------|-----------|
| **Convex Models** (Linear, Logistic, SVM, Perceptron) | **SGD** or **SGD + Momentum** | Simple, efficient for convex optimization |
| **Deep Models** (MLP, CNN, RNN) | **Adam** or **RMSProp** | Better handling of complex loss landscapes |
| **Non-gradient Models** (kNN, Naive Bayes, Trees, Random Forest) | **None** | No iterative optimization required |