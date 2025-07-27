#include "models/mlp.hpp"
#include "common.hpp"
#include <iostream>
#include <fstream>
#include <sstream>

bool loadMNISTCSV(const std::string &filename,
  std::vector<std::vector<float>> &X,
  std::vector<uint8_t> &y,
  int maxSamples = -1) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error: Could not open file " << filename << std::endl;
    return false;
  }
  std::string line;
  std::getline(file, line); // Skip header if present
  int sampleCount = 0;
  while (std::getline(file, line) && (maxSamples == -1 || sampleCount < maxSamples)) {
    std::stringstream ss(line);
    std::string cell;
    // First column is label
    std::getline(ss, cell, ',');
    int label = std::stoi(cell);
    y.push_back(label);
    // Rest are pixel values
    std::vector<float> features;
    while (std::getline(ss, cell, ',')) {
      float pixel = std::stof(cell) / 255.0f; // Normalize to [0,1]
      features.push_back(pixel);
    }
    X.push_back(features);
    sampleCount++;
  }
  std::cout << "Loaded " << X.size() << " samples with " << X[0].size() << " features each" << std::endl;
  return true;
}

// One-hot encode label (e.g., 3 -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
std::vector<float> one_hot(uint8_t label, size_t num_classes = 10) {
  std::vector<float> encoded(num_classes, 0.0f);
  encoded[label] = 1.0f;
  return encoded;
}

float evaluate_accuracy(models::MLP &model,
  const std::vector<std::vector<float>> &X,
  const std::vector<uint8_t> &y) {
  size_t correct = 0;
  for (size_t i = 0; i < X.size(); ++i) {
    auto output = model.predict(X[i]);
    size_t predicted = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
    if (predicted == y[i]) ++correct;

    //std::cout << "Sample " << i << ": Predicted = " << predicted
    //  << ", Actual = " << y[i]
    //  << ", Confidence = " << std::setprecision(3) << output[predicted] * 100 << "%" << std::endl;

  }
  return static_cast<float>(correct) / X.size();
}

void test_mnist_mlp() {
  std::cout << "[MNIST MLP]" << std::endl;

  std::vector<uint8_t> train_labels;
  std::vector<uint8_t> test_labels;

  std::vector<std::vector<float>> X_train, X_test;
  std::vector<int> y_test;

  if (!loadMNISTCSV(utils::userDir() + "/workspace/misc/data/mnist_train.csv", X_train, train_labels, 5000)) { // Use subset for faster training
    std::cout << "Error loading training data. Please download MNIST CSV files." << std::endl;
    std::cout << "You can get them from: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv" << std::endl;
    return;
  }

  if (!loadMNISTCSV(utils::userDir() + "/workspace/misc/data/mnist_test.csv", X_test, test_labels, 1000)) { // Use subset for faster testing
    std::cout << "Error loading test data." << std::endl;
    return;
  }

  std::vector<std::vector<float>> y_train;
  for (uint8_t label : train_labels) {
    y_train.push_back({ one_hot(label) });
  }

  models::MLP mlp({ 784, 128, 64, 10 }, 0.01f, models::MLP::Initialization::RandomUniform);
  mlp.fit(X_train, y_train, 20, models::MLP::ActivationF::Softmax);

  float accuracy = evaluate_accuracy(mlp, X_test, test_labels);
  std::cout << "Test Accuracy: " << accuracy * 100.0f << "%\n";
  std::cout << "[MNIST Test] " << (accuracy >= 0.85f ? "PASSED" : "LOW ACCURACY") << "\n";
}
