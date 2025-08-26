#include "models/rnn.hpp"
#include <iostream>
#include <vector>

void test_rnn_forward() {
  // Create a toy RNN: input_size=2, hidden_size=4, output_size=2
  models::RNN rnn(2, 4, 2);

  // Example input sequence: 3 timesteps, each input has 2 features
  std::vector<std::vector<float>> sequence = {
      {0.5f, -0.2f},
      {0.1f,  0.9f},
      {-0.7f, 0.3f}
  };

  // Forward pass
  auto outputs = rnn.predict(sequence);

  std::cout << "[RNN] Basic forward test: ";

  // Check: number of outputs should equal number of timesteps
  if (outputs.size() != sequence.size()) {
    std::cout << "FAILED: Output length mismatch. Got "
      << outputs.size() << " expected " << sequence.size() << "\n";
    return;
  }

  // Check: each output vector should have size = output_size (2 here)
  bool correct_shape = true;
  for (auto &y : outputs) {
    if (y.size() != 2) {
      correct_shape = false;
      break;
    }
  }

  if (!correct_shape) {
    std::cout << "FAILED (Each output must have size 2.)\n";
    return;
  }

  std::cout << "PASSED\n";
}

void test_rnn() {
  test_rnn_forward();
  models::RNN::run_basic_test();

  std::cout << std::fixed << std::setprecision(3);

#if 1
  // Test 1: Simple Pattern Recognition - Learn basic temporal patterns
  {
    //std::cout << "Test 1: Simple Temporal Pattern Recognition" << std::endl;
    //std::cout << "-------------------------------------------" << std::endl;

    models::RNN rnn(1, 10, 1);  // 1 input, 10 hidden, 1 output

    // Create simple temporal patterns that are more suitable for RNNs
    // Pattern: If input > 0.5, output 1.0, else output 0.0 (delayed by one step)
    std::vector<std::vector<std::vector<float>>> train_sequences;
    std::vector<std::vector<std::vector<float>>> train_targets;

    // Generate training sequences of length 3
    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (int seq = 0; seq < 200; ++seq) {
      std::vector<std::vector<float>> sequence;
      std::vector<std::vector<float>> target;

      float prev_input = 0.0f;  // Start with 0
      for (int t = 0; t < 3; ++t) {
        float current_input = (dis(gen) > 0.5f) ? 1.0f : 0.0f;  // Binary inputs
        sequence.push_back({ current_input });

        // Target is previous input (delayed echo)
        target.push_back({ prev_input });
        prev_input = current_input;
      }

      train_sequences.push_back(sequence);
      train_targets.push_back(target);
    }

    // Train the network
    //std::cout << "Training on delayed echo task (200 sequences, 500 epochs)..." << std::endl;
    rnn.fit(train_sequences, train_targets, 500, 0.05f, models::ActivationF::Tanh, models::ActivationF::Sigmoid);

    // Test on simple patterns
    //std::cout << "Delayed Echo Test Results:" << std::endl;
    std::vector<std::vector<std::vector<float>>> test_sequences = {
      {{0.0f}, {0.0f}, {0.0f}},  // Should output: 0, 0, 0
      {{0.0f}, {0.0f}, {1.0f}},  // Should output: 0, 0, 0  
      {{0.0f}, {1.0f}, {0.0f}},  // Should output: 0, 0, 1
      {{1.0f}, {0.0f}, {0.0f}},  // Should output: 0, 1, 0
      {{1.0f}, {1.0f}, {1.0f}}   // Should output: 0, 1, 1
    };

    std::vector<std::vector<std::vector<float>>> test_targets = {
      {{0.0f}, {0.0f}, {0.0f}},
      {{0.0f}, {0.0f}, {0.0f}},
      {{0.0f}, {0.0f}, {1.0f}},
      {{0.0f}, {1.0f}, {0.0f}},
      {{0.0f}, {1.0f}, {1.0f}}
    };

    int correct_predictions = 0;
    int total_predictions = 0;

    for (size_t i = 0; i < test_sequences.size(); ++i) {
      auto prediction = rnn.predict(test_sequences[i]);

      //std::cout << "  Input: [";
      //for (const auto &step : test_sequences[i]) {
      //  std::cout << step[0] << " ";
      //}
      //std::cout << "] -> Expected: [";
      //for (const auto &step : test_targets[i]) {
      //  std::cout << step[0] << " ";
      //}
      //std::cout << "], Predicted: [";

      for (size_t t = 0; t < prediction.size(); ++t) {
        float pred_val = prediction[t][0];
        float expected_val = test_targets[i][t][0];
        //std::cout << std::fixed << std::setprecision(2) << pred_val << " ";

        // Check if prediction is close to target (within 0.3)
        if (std::abs(pred_val - expected_val) < 0.3f) {
          correct_predictions++;
        }
        total_predictions++;
      }
      //std::cout << "]" << std::endl;
    }

    float accuracy = static_cast<float>(correct_predictions) / total_predictions * 100.0f;
    std::cout << "[RNN] Delayed echo test: " << (75 < accuracy ? "PASSED" : "FAILED")
      << " (" << correct_predictions << "/" << total_predictions << " correct, " << accuracy << "%)\n";
  }
#endif

#if 0
  // Test 2: Memory Task
  {
    std::cout << "Test 2: Sequence Memory (Remember First Input)" << std::endl;
    std::cout << "----------------------------------------------" << std::endl;


    std::vector<std::vector<std::vector<float>>> train_sequences;
    std::vector<std::vector<std::vector<float>>> train_targets;

    std::random_device rd;
    std::mt19937 gen(/*42*/rd());  // Fixed seed
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (int i = 0; i < 50; ++i) {
      float first_val = dis(gen);
      std::vector<std::vector<float>> sequence = {
          {first_val}, {dis(gen)}, {dis(gen)}
      };
      std::vector<std::vector<float>> target = {
          {0.0f}, {0.0f}, {first_val}
      };
      train_sequences.push_back(sequence);
      train_targets.push_back(target);
    }

    models::RNN rnn(1, 10, 1);
    rnn.fit(train_sequences, train_targets, 300, 0.02f, models::ActivationF::Tanh, models::ActivationF::Sigmoid);

    // Quick test
    float test_vals[] = { 0.2f, 0.7f, 0.9f };
    for (float test_val : test_vals) {
      std::vector<std::vector<float>> test_seq = {
          {test_val}, {0.5f}, {0.3f}
      };
      auto pred = rnn.predict(test_seq);
      float predicted_memory = pred[2][0];
      std::cout << "  Remember " << test_val << " -> "
        << std::fixed << std::setprecision(3) << predicted_memory
        << " (Error: " << std::abs(test_val - predicted_memory) << ")" << std::endl;
    }
  }
#endif

#if 1
  // Test 3: Simple Classification
  {
    //std::cout << "\nTest 3: Simple Binary Classification" << std::endl;
    //std::cout << "------------------------------------" << std::endl;

    models::RNN rnn(1, 8, 2);  // Simplified: just classify as "mostly 0s" or "mostly 1s"

    std::vector<std::vector<std::vector<float>>> train_sequences;
    std::vector<std::vector<std::vector<float>>> train_targets;

    // Generate sequences and classify based on majority
    for (int seq = 0; seq < 100; ++seq) {
      std::vector<std::vector<float>> sequence;
      int ones_count = 0;

      // Random 3-step binary sequence
      for (int t = 0; t < 3; ++t) {
        float val = (rand() % 2 == 0) ? 0.0f : 1.0f;
        sequence.push_back({ val });
        if (val > 0.5f) ones_count++;
      }

      // Target: [1,0] if mostly 0s, [0,1] if mostly 1s
      std::vector<std::vector<float>> target(3);
      if (ones_count <= 1) {
        target = { {1.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 0.0f} };  // Mostly 0s
      } else {
        target = { {0.0f, 1.0f}, {0.0f, 1.0f}, {0.0f, 1.0f} };  // Mostly 1s
      }

      train_sequences.push_back(sequence);
      train_targets.push_back(target);
    }

    //std::cout << "Training binary classifier..." << std::endl;
    rnn.fit(train_sequences, train_targets, 200, 0.05f,
      models::ActivationF::Tanh, models::ActivationF::Softmax);

    // Test
    std::vector<std::vector<float>> test_seqs[] = {
        {{0}, {0}, {0}},  // Should classify as "mostly 0s"
        {{1}, {1}, {1}},  // Should classify as "mostly 1s" 
        {{1}, {0}, {0}}   // Should classify as "mostly 0s"
    };

    std::vector<int> test_labels = { 0, 1, 0 };  // Expected classes
    int correct_predictions = 0;
    for (int i = 0; i < 3; ++i) {
      auto pred = rnn.predict(test_seqs[i]);
      auto final_output = pred[2];
      int predicted_class = (final_output[0] > final_output[1]) ? 0 : 1;
      if (predicted_class == test_labels[i]) {
        correct_predictions++;
      }

      //std::cout << "  Sequence [";
      //for (auto &step : test_seqs[i]) std::cout << (int)step[0] << " ";
      //std::cout << "] -> Class " << predicted_class
      //  << " (confidence: " << std::max(final_output[0], final_output[1]) << ")" << std::endl;
    }
    float accuracy = static_cast<float>(correct_predictions) / 3 * 100.0f;
    std::cout << "[RNN] Simple binary classification: "
      << (accuracy >= 66.6f ? "PASSED" : "FAILED")
      << " (" << correct_predictions << "/3 correct, " << accuracy << "%)\n";
  }
#endif

#if 1
  // Test 4: Time Series Prediction - Sine Wave
  {
    //std::cout << "Test 4: Time Series Prediction (Sine Wave)" << std::endl;
    //std::cout << "------------------------------------------" << std::endl;


    std::vector<std::vector<std::vector<float>>> train_sequences;
    std::vector<std::vector<std::vector<float>>> train_targets;

    // Generate sine wave training data
    const int sequence_length = 5;
    const int num_sequences = 100;

    for (int seq = 0; seq < num_sequences; ++seq) {
      float start_t = static_cast<float>(seq) * 0.1f;
      std::vector<std::vector<float>> sequence;
      std::vector<std::vector<float>> target;

      for (int t = 0; t < sequence_length; ++t) {
        float time = start_t + t * 0.2f;
        float current_val = std::sin(time);
        float next_val = std::sin(time + 0.2f);

        sequence.push_back({ current_val });
        target.push_back({ next_val });
      }

      train_sequences.push_back(sequence);
      train_targets.push_back(target);
    }

    // Train
    //std::cout << "Training sine wave predictor (100 sequences, 400 epochs)..." << std::endl;
    models::RNN rnn(1, 15, 1, false);  // 1 input, 15 hidden, 1 output
    rnn.initialize_weights(models::Initialization::Xavier);
    rnn.fit(train_sequences, train_targets, 800, 0.005f, models::ActivationF::Tanh, models::ActivationF::Tanh);  // Tanh for both since sine is [-1,1]

    // Test prediction
    //std::cout << "Sine Wave Prediction Test:" << std::endl;
    float test_start = 10.0f;  // Use unseen time range
    std::vector<std::vector<float>> test_sequence;
    std::vector<float> expected_next;

    for (int t = 0; t < sequence_length; ++t) {
      float time = test_start + t * 0.2f;
      float val = std::sin(time);
      test_sequence.push_back({ val });
      expected_next.push_back(std::sin(time + 0.2f));
    }

    auto predictions = rnn.predict(test_sequence);

    //std::cout << "  Time step predictions:" << std::endl;
    float total_error = 0.0f;
    for (int t = 0; t < sequence_length; ++t) {
      float predicted = predictions[t][0];
      float expected = expected_next[t];
      float error = std::abs(predicted - expected);
      total_error += error;

      //std::cout << "    t=" << t << ": sin(" << std::fixed << std::setprecision(2)
      //  << (test_start + t * 0.2f + 0.2f) << ") = " << std::setprecision(4)
      //  << expected << ", Predicted: " << predicted
      //  << " (Error: " << error << ")" << std::endl;
    }
    float err = total_error / sequence_length;
    std::cout << "[RNN] Sine wave prediction: " << (err < 0.1 ? "PASSED" : "FAILED") << " (avg error " << err << ")\n";
  }
#endif

}
