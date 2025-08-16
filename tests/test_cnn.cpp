// tests/test_cnn.cpp
#include "models/cnn.hpp"
#include "common.hpp"
#include "datasynth.hpp"
#include <cassert>
#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>


void print_result(bool passed, const std::string &test_name) {
  std::cout << test_name << (passed ? " PASSED" : " FAILED") << std::endl;
}

void test_cnn_conv() {
  // Dummy 1x5x5 input - CORRECTED: properly structured as [channels][height][width]
  std::vector<std::vector<std::vector<float>>> input = {
    {  // Channel 0
      {1, 2, 3, 0, 1},  // Row 0
      {0, 1, 2, 3, 1},  // Row 1  
      {1, 0, 1, 2, 2},  // Row 2
      {2, 1, 0, 1, 0},  // Row 3
      {1, 2, 3, 2, 1}   // Row 4
    }
  };

  // Dummy 1 filter of 3x3 - this looks correct
  std::vector<std::vector<std::vector<std::vector<float>>>> filters = {
    {  // Output channel 0
      {  // Input channel 0
        {1, 0, -1},   // Kernel row 0
        {1, 0, -1},   // Kernel row 1  
        {1, 0, -1}    // Kernel row 2
      }
    }
  };

  models::Conv2D conv(1, 1, 3, 1, 0); // 1 input, 1 output channel, 3x3, stride=1, no padding
  conv.setWeights(filters);
  auto conv_output = conv.forward(input);

  // Expected output size: (5-3+0)/1 + 1 = 3x3
  assert(conv_output.size() == 1);           // 1 output channel
  assert(conv_output[0].size() == 3);        // 3 rows
  assert(conv_output[0][0].size() == 3);     // 3 columns


  print_result(conv_output[0][0].size() == 3, "[Conv2D]");

  //models::MaxPool2D pool(2);
  //auto pooled_output = pool.forward(conv_output);
  //print_result(pooled_output[0][0].size() == 1 * 1, "[MaxPool2D]");
}

void test_cnn_maxpool() {
  // 1x4x4 input
  std::vector<std::vector<std::vector<float>>> input = {
    {
      {1, 2, 3, 4},
      {5, 6, 7, 8},
      {9, 10, 11, 12},
      {13, 14, 15, 16}
    }
  };

  models::MaxPool2D pool(2, 2); // 2x2 pooling, stride=2
  auto output = pool.forward(input);

  // Expected: 2x2 output with values [6, 8, 14, 16]
  if (output[0].size() == 2 && output[0][0].size() == 2) {
    assert(output[0][0][0] == 6);
    assert(output[0][0][1] == 8);
    assert(output[0][1][0] == 14);
    assert(output[0][1][1] == 16);
    std::cout << "[MaxPool] PASSED" << std::endl;
  } else {
    std::cout << "[MaxPool] FAILED" << std::endl;
  }
}


void test_cnn_basic() {
  //std::cout << "=== Testing CNN Basic Functionality ===" << std::endl;

  // Create a simple CNN for 28x28 grayscale images (like MNIST)
  // Input: 1 channel, 28x28
  // Final feature size: 16 channels * 4 * 4 = 256
  // Set up MLP: 256 -> 120 -> 84 -> 10 (for 10-class classification)
  models::CNN cnn(1, { 256, 120, 84, 10 }, 0.1f);

  // Add first conv layer: 1->6 channels, 5x5 kernel
  // Output size: (28-5+0)/1 + 1 = 24x24
  cnn.add_conv_layer(6, 5, 1, 0);  // stride=1, padding=0
  cnn.set_conv_weights(0, utils::createRandomWeightsForConvLayer(6, 1, 5));

  // Add first pooling layer: 2x2 max pooling
  // Output size: 24/2 = 12x12
  cnn.add_pool_layer(2, 2);  // pool_size=2, stride=2

  // Add second conv layer: 6->16 channels, 5x5 kernel
  // Output size: (12-5+0)/1 + 1 = 8x8
  cnn.add_conv_layer(16, 5, 1, 0);
  cnn.set_conv_weights(1, utils::createRandomWeightsForConvLayer(16, 6, 5));

  // Add second pooling layer: 2x2 max pooling
  // Output size: 8/2 = 4x4
  cnn.add_pool_layer(2, 2);


  // Create test input: 1 channel, 28x28 image
  std::vector<std::vector<std::vector<float>>> test_input(1);
  test_input[0].resize(28);
  for (int i = 0; i < 28; ++i) {
    test_input[0][i].resize(28);
    for (int j = 0; j < 28; ++j) {
      // Simple gradient pattern
      test_input[0][i][j] = (float)(i + j) / 56.0f;
    }
  }

  //std::cout << "Input shape: " << test_input.size() << "x" << test_input[0].size() << "x" << test_input[0][0].size() << std::endl;

  // Test forward pass
  //std::cout << "Running forward pass..." << std::endl;
  std::vector<float> output = cnn.forward(test_input);

  //std::cout << "Output size: " << output.size() << std::endl;
  //assert(output.size() == 10 && "Output should have 10 elements");

  //std::cout << "Output values: ";
  //for (size_t i = 0; i < output.size(); ++i) {
  //  std::cout << std::fixed << std::setprecision(4) << output[i];
  //  if (i < output.size() - 1) std::cout << ", ";
  //}
  //std::cout << std::endl;

  // Test prediction
  int predicted_class = cnn.predict_class(test_input);
  //std::cout << "Predicted class: " << predicted_class << std::endl;
  //assert(predicted_class >= 0 && predicted_class < 10 && "Predicted class should be between 0-9");

  //std::cout << "Basic CNN test passed!" << std::endl;
  //std::cout << "\n";

  std::cout << "[CNN] Basic test: " << (output.size() == 10 && 0 <= predicted_class && predicted_class < 10 ? "PASSED" : "FAILED") << std::endl;
}

void test_cnn_small() {
  //std::cout << "=== Testing CNN with Small Input ===" << std::endl;

  // Create a tiny CNN for testing
  // MLP: 8*4*4 = 128 -> 32 -> 3
  models::CNN cnn(3, { 128, 32, 3 }, 0.01f);  // 3 channels (RGB)

  // Add one conv layer: 3->8 channels, 3x3 kernel, padding=1
  // For 8x8 input with padding=1: output = 8x8
  cnn.add_conv_layer(8, 3, 1, 1);
  cnn.set_conv_weights(0, utils::createRandomWeightsForConvLayer(8, 3, 3));

  // Add pooling: 2x2
  // Output: 4x4
  cnn.add_pool_layer(2, 2);

  // Create 8x8 RGB test input
  std::vector<std::vector<std::vector<float>>> test_input(3);
  for (int c = 0; c < 3; ++c) {
    test_input[c].resize(8);
    for (int i = 0; i < 8; ++i) {
      test_input[c][i].resize(8);
      for (int j = 0; j < 8; ++j) {
        test_input[c][i][j] = (float)(c * 64 + i * 8 + j) / 192.0f;
      }
    }
  }

  //std::cout << "Small input shape: " << test_input.size() << "x" << test_input[0].size() << "x" << test_input[0][0].size() << std::endl;

  std::vector<float> output = cnn.forward(test_input);
  //std::cout << "Small output size: " << output.size() << std::endl;
  //assert(output.size() == 3 && "Output should have 3 elements");

  int predicted = cnn.predict_class(test_input);
  //std::cout << "Small predicted class: " << predicted << std::endl;
  //assert(predicted >= 0 && predicted < 3 && "Predicted class should be between 0-2");

  //std::cout << "Small CNN test passed!" << std::endl;
  //std::cout << "\n";

  std::cout << "[CNN] Sanity test: " << (output.size() == 3 && 0 <= predicted && predicted < 3 ? "PASSED" : "FAILED") << std::endl;
}

void test_cnn_image_data() {
  //std::cout << "=== Testing CNN on Real Data ===" << std::endl;

    // Option 1: Load actual BMP images (if you have them)
    /*
    BMPLoader::BMPImage img1 = BMPLoader::load_bmp("test_image1.bmp");
    BMPLoader::BMPImage resized = BMPLoader::resize(img1, 32, 32);
    auto grayscale = BMPLoader::to_grayscale(resized);
    */

    // Option 2: Use synthetic dataset for testing
  std::vector<std::vector<std::vector<std::vector<float>>>> all_images;
  std::vector<int> all_labels;
  generate_grayscale_images(all_images, all_labels, 200);

  // Convert labels to one-hot encoding for MLP
  std::vector<std::vector<float>> all_labels_onehot(all_labels.size());
  for (size_t i = 0; i < all_labels.size(); ++i) {
    all_labels_onehot[i].resize(3, 0.0f); // 3 classes
    all_labels_onehot[i][all_labels[i]] = 1.0f;
  }

  // Create CNN for 32x32 grayscale images, 3 classes
  // Final size: 16 * 8 * 8 = 1024
  models::CNN cnn(1, { 1024, 128, 64, 3 }, 0.1f);
  cnn.add_conv_layer(8, 5, 1, 2);   // 32x32 -> 32x32 (with padding)
  cnn.add_pool_layer(2, 2);         // 32x32 -> 16x16
  cnn.add_conv_layer(16, 3, 1, 1);  // 16x16 -> 16x16 (with padding)
  cnn.add_pool_layer(2, 2);         // 16x16 -> 8x8

  // Initialize weights
  cnn.set_conv_weights(0, utils::createXavierWeightsForConvLayer(8, 1, 5));
  cnn.set_conv_weights(1, utils::createXavierWeightsForConvLayer(16, 8, 3));


  // Split data: 80% train, 20% test
  size_t train_size = (all_images.size() * 80) / 100;

  std::vector<std::vector<std::vector<std::vector<float>>>> train_images(
    all_images.begin(), all_images.begin() + train_size);
  std::vector<std::vector<float>> train_labels(
    all_labels_onehot.begin(), all_labels_onehot.begin() + train_size);

  std::vector<std::vector<std::vector<std::vector<float>>>> test_images(
    all_images.begin() + train_size, all_images.end());
  std::vector<int> test_labels_int(
    all_labels.begin() + train_size, all_labels.end());

  //std::cout << "Train set: " << train_images.size() << " images" << std::endl;
  //std::cout << "Test set: " << test_images.size() << " images" << std::endl;


  // TRAIN THE CNN
  //std::cout << "\n--- TRAINING PHASE ---" << std::endl;
  cnn.train(train_images, train_labels, 100, models::MLP::ActivationF::Softmax);


  // Test on several images
  int correct_predictions = 0;
  std::vector<std::vector<int>> confusion_matrix(3, std::vector<int>(3, 0));
  const auto testCount = std::min(test_images.size(), size_t(50));
  for (size_t i = 0; i < testCount; ++i) {
    auto output = cnn.forward(test_images[i]);
    int predicted = cnn.predict_class(test_images[i]);
    int actual = test_labels_int[i];

    confusion_matrix[actual][predicted]++;

    if (predicted == actual) {
      correct_predictions++;
    }
    //if (i < 10) { // Show first 10 predictions
    //  std::cout << "Test " << i << ": Predicted=" << predicted
    //    << " (conf=" << std::fixed << std::setprecision(3) << output[predicted]
    //    << "), Actual=" << actual
    //    << (predicted == actual ? " GOOD" : " BAD") << std::endl;
    //}
  }

  // Print results
  //std::cout << "\n=== Results ===" << std::endl;
  //std::cout << "Accuracy: " << correct_predictions << "/20 = " << (100.0 * correct_predictions / 20) << "%" << std::endl;

  //std::cout << "\nConfusion Matrix:" << std::endl;
  //std::cout << "Actual\\Predicted\t0\t1\t2" << std::endl;
  //for (int i = 0; i < 3; ++i) {
  //  std::cout << i << "\t\t\t";
  //  for (int j = 0; j < 3; ++j) {
  //    std::cout << confusion_matrix[i][j] << "\t";
  //  }
  //  std::cout << std::endl;
  //}


  // Test individual class performance
  //std::cout << "\nClass Performance:" << std::endl;
  //for (int i = 0; i < 3; ++i) {
  //  int total_actual = 0;
  //  for (int j = 0; j < 3; ++j) {
  //    total_actual += confusion_matrix[i][j];
  //  }
  //  if (total_actual > 0) {
  //    float class_accuracy = 100.0f * confusion_matrix[i][i] / total_actual;
  //    std::cout << "Class " << i << ": " << std::fixed << std::setprecision(1)
  //      << class_accuracy << "% (" << confusion_matrix[i][i]
  //      << "/" << total_actual << ")" << std::endl;
  //  }
  //}
  //std::cout << "\n";

  std::cout << "[CNN] 32x32 image data test: " 
            << (0.7 < (correct_predictions/float(testCount)) ? "PASSED" : "FAILED")
    << " (" << correct_predictions << "/" << testCount << " correct)" << std::endl;
}

#if 0
void debug_cnn_step_by_step() { // SUCCEEDS
  std::cout << "=== STEP-BY-STEP CNN DEBUG ===" << std::endl;

  // Create ONE simple test image
  std::vector<std::vector<std::vector<float>>> test_image(1);
  test_image[0].resize(32);
  for (int y = 0; y < 32; ++y) {
    test_image[0][y].resize(32);
    for (int x = 0; x < 32; ++x) {
      // Simple horizontal stripe pattern (class 0)
      test_image[0][y][x] = (y / 4) % 2 == 0 ? 0.8f : 0.2f;
    }
  }

  std::cout << "Created test image 32x32" << std::endl;

  // Create CNN
  models::CNN cnn(1, { 1024, 32, 3 }, 0.1f);
  cnn.add_conv_layer(4, 3, 1, 1);
  cnn.add_pool_layer(2, 2);

  // Calculate expected feature size
  // Input: 32x32
  // After conv (3x3, stride=1, padding=1): 32x32, 4 channels
  // After pool (2x2, stride=2): 16x16, 4 channels
  // Flattened: 4 * 16 * 16 = 1024
  std::cout << "Expected feature size: 1024" << std::endl;

  // Initialize weights
  cnn.set_conv_weights(0, utils::createXavierWeightsForConvLayer(4, 1, 3));

  // STEP 1: Test feature extraction
  std::cout << "\nSTEP 1: Testing feature extraction..." << std::endl;
  try {
    auto features = cnn.extract_features(test_image);
    std::cout << "Feature extraction successful" << std::endl;
    std::cout << "Feature vector size: " << features.size() << std::endl;

    // Check feature statistics
    float min_val = *std::min_element(features.begin(), features.end());
    float max_val = *std::max_element(features.begin(), features.end());
    float sum = std::accumulate(features.begin(), features.end(), 0.0f);
    float mean = sum / features.size();

    std::cout << "Feature stats - Min: " << min_val << ", Max: " << max_val << ", Mean: " << mean << std::endl;

    // Count zeros
    int zero_count = std::count(features.begin(), features.end(), 0.0f);
    std::cout << "Zero features: " << zero_count << "/" << features.size() << std::endl;

    // Check for NaN/Inf
    bool has_bad_values = false;
    for (float f : features) {
      if (std::isnan(f) || std::isinf(f)) {
        has_bad_values = true;
        break;
      }
    }
    std::cout << "Has NaN/Inf: " << (has_bad_values ? "YES - PROBLEM!" : "No") << std::endl;

  } catch (const std::exception &e) {
    std::cout << " Feature extraction failed: " << e.what() << std::endl;
    return;
  }

  // STEP 2: Test MLP creation and parameters
  std::cout << "\nSTEP 2: Checking MLP construction..." << std::endl;

  // Create a separate MLP with same parameters as CNN uses
  try {
    models::MLP test_mlp({ 1024, 32, 3 }, 0.1f, models::MLP::Initialization::Xavier);
    std::cout << " MLP construction successful" << std::endl;

    // Test prediction on random features
    std::vector<float> random_features(1024);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0f, 0.1f);
    for (float &f : random_features) {
      f = dis(gen);
    }

    auto test_output = test_mlp.predict(random_features);
    std::cout << "MLP output size: " << test_output.size() << std::endl;
    std::cout << "MLP output: [" << test_output[0] << ", " << test_output[1] << ", " << test_output[2] << "]" << std::endl;

  } catch (const std::exception &e) {
    std::cout << " MLP test failed: " << e.what() << std::endl;
    return;
  }

  // STEP 3: Test CNN training with minimal data
  std::cout << "\nSTEP 3: Testing CNN training with minimal data..." << std::endl;

  // Create 3 training samples (one per class)
  std::vector<std::vector<std::vector<std::vector<float>>>> train_data(3);
  std::vector<std::vector<float>> train_labels(3);

  for (int class_id = 0; class_id < 3; ++class_id) {
    train_data[class_id].resize(1); // 1 channel
    train_data[class_id][0].resize(32);

    // One-hot label
    train_labels[class_id].resize(3, 0.0f);
    train_labels[class_id][class_id] = 1.0f;

    for (int y = 0; y < 32; ++y) {
      train_data[class_id][0][y].resize(32);
      for (int x = 0; x < 32; ++x) {
        float value = 0.0f;
        switch (class_id) {
        case 0: value = (y / 4) % 2 == 0 ? 0.8f : 0.2f; break; // Horizontal
        case 1: value = (x / 4) % 2 == 0 ? 0.8f : 0.2f; break; // Vertical
        case 2: value = ((x / 4) + (y / 4)) % 2 == 0 ? 0.8f : 0.2f; break; // Checkerboard
        }
        train_data[class_id][0][y][x] = value;
      }
    }
  }

  std::cout << "Created 3 training samples" << std::endl;

  // Extract features for each training sample BEFORE training
  std::cout << "Features before training:" << std::endl;
  for (int i = 0; i < 3; ++i) {
    auto features = cnn.extract_features(train_data[i]);
    float mean = std::accumulate(features.begin(), features.end(), 0.0f) / features.size();
    std::cout << "Sample " << i << " feature mean: " << mean << std::endl;
  }

  // Test predictions BEFORE training
  std::cout << "Predictions before training:" << std::endl;
  for (int i = 0; i < 3; ++i) {
    int pred = cnn.predict_class(train_data[i]);
    auto output = cnn.forward(train_data[i]);
    std::cout << "Sample " << i << " (class " << i << "): predicted=" << pred
      << ", outputs=[" << output[0] << ", " << output[1] << ", " << output[2] << "]" << std::endl;
  }

  // NOW TRAIN
  std::cout << "\nTraining CNN..." << std::endl;
  try {
    cnn.train(train_data, train_labels, 20, models::MLP::ActivationF::Sigmoid);
    std::cout << " Training completed" << std::endl;
  } catch (const std::exception &e) {
    std::cout << " Training failed: " << e.what() << std::endl;
    return;
  }

  // Test predictions AFTER training
  std::cout << "Predictions after training:" << std::endl;
  int correct = 0;
  for (int i = 0; i < 3; ++i) {
    int pred = cnn.predict_class(train_data[i]);
    auto output = cnn.forward(train_data[i]);
    std::cout << "Sample " << i << " (class " << i << "): predicted=" << pred
      << ", outputs=[" << std::fixed << std::setprecision(3)
      << output[0] << ", " << output[1] << ", " << output[2] << "]" << std::endl;
    if (pred == i) correct++;
  }

  std::cout << "\nTraining accuracy: " << correct << "/3 = " << (100.0 * correct / 3) << "%" << std::endl;

  if (correct == 0) {
    std::cout << "PROBLEM: Still 0% accuracy after training!" << std::endl;
  } else {
    std::cout << "SUCCESS: Training is working!" << std::endl;
  }
}
#endif

void test_cnn() {
  test_cnn_conv();
  test_cnn_maxpool();
  std::cout << "\n";
  test_cnn_basic();
  test_cnn_small();
  test_cnn_image_data();
  //debug_cnn_step_by_step();
}