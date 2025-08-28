# AI From Scratch in C++

> Pure C++ implementations of popular AI and machine learning algorithms without using any third-party libraries.
> No Python, no frameworks, no third-party libraries — just clean, idiomatic C++.

---

## Weekly Schedule

| Week | Topic                        | Key Concepts |
|------|-----------------------------|--------------|
| 1    | Linear Regression            | MSE Loss, Gradient Descent, fit/predict, synthetic data |
| 2    | Logistic Regression          | Sigmoid, Binary Cross-Entropy, binary classification |
| 3    | k-Nearest Neighbors          | Euclidean distance, memory, sorting, majority voting |
| 4    | Naive Bayes (Gaussian)       | Class priors, Gaussian PDF, log prob math |
| 5    | Decision Tree                | Entropy, Information Gain, recursive tree prediction |
| 6    | Random Forest                | Bootstrap sampling, ensemble voting |
| 7    | Support Vector Machine       | Hard-margin linear SVM, max-margin hyperplane |
| 8    | Matrix Math Library          | Dot products, transpose, scalar ops, operator overloading |
| 9    | Perceptron                   | Basic neural unit, step/tanh, logic gates training |
| 10   | Multilayer Perceptron (MLP)  | Feedforward, backpropagation, ReLU/sigmoid |
| 11   | MNIST with MLP               | PGM parsing, feedforward classification |
| 12   | Convolutional Neural Networks| 2D conv, pooling, stride, combine with MLP |
| 13   | Recurrent Neural Networks    | Time sequences, memory state, BPTT |
| 14   | Optimizers                   | SGD, Momentum, AdaGrad, RMSProp, Adam |
| 15   | Regularization Techniques    | L1/L2, Dropout, Early stopping |
| 16   | Reinforcement Learning       | Q-learning, gridworld, Q-function, Bellman update rule |
| 17   | Unsupervised Learning        | k-Means clustering, PCA - Principal Component Analysis |
| 18   | Genetic Algorithms           | Evolutionary algorithms |
| 19   | TO BE DECIDED                |  |
| 20   | Final Projects               | Choose from CNN, chatbot, RL agent, etc |

---

## Structure for Each Week

- Theory Markdown (`weekX_<topic>.md`)
- C++ 20 Implementation:
  - `src/models/<topic>.hpp/.cpp`
  - `tests/test_<topic>.cpp` with PASSED/FAILED outputs
- Uses SFML for optional visualization of some of the models like Linear Regression.
- Ephasis on intuition, clarity, and systems-level understanding

---

## Final Outcome

By the end of the course, you’ll have:
- Your own mini ML library in C++
- Deep understanding of classic AI/ML algorithms
- Practical experience in building from scratch
- Reusable, testable, extendable codebase

---

```
mkdir build
cd build

cmake ...

make
```
