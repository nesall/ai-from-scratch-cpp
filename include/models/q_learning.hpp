#pragma once
#include <vector>
#include <utility>
#include <random>
#include "matrix.hpp"

// Reinforcement learning.

namespace utils {

  class Gridworld {
  public:
    enum Action { UP, DOWN, LEFT, RIGHT };

    Gridworld(int width, int height, std::pair<int, int> start,
      std::pair<int, int> goal,
      const std::vector<std::pair<int, int>> &traps);

    void reset();
    std::pair<int, int> getState() const;
    bool isTerminal(const std::pair<int, int> &state) const;

    // Step environment with an action
    // Returns: (next_state, reward, done)
    std::tuple<std::pair<int, int>, float, bool> step(Action action);

    int state2index(const std::pair<int, int> &state) const;
    int nofStates() const { return width_ * height_; }
    int nofActions() const { return 4; }

    int rows() const { return height_; }
    int cols() const { return width_; }

    std::pair<int, int> start() const { return start_; }
    std::vector<std::pair<int, int>> traps() const { return traps_; }
    std::pair<int, int> goal() const { return goal_; }

  private:
    int width_;
    int height_;
    std::pair<int, int> start_;
    std::pair<int, int> goal_;
    std::vector<std::pair<int, int>> traps_;

    std::pair<int, int> state_; // current agent position
  };

} // namespace utils


namespace models {

  class Q_LearningAgent {
  public:
    Q_LearningAgent(int nstates, int nactions, float alpha = 0.1f, float gamma = 0.99f, float epsilon = 0.2f);

    // Choose action based on epsilon-greedy policy
    int selectAction(int state);

    // Update Q-values based on Bellman equation
    void update(int state, int action, float reward, int nextState, bool done);

    // Decay epsilon over time
    void decayEpsilon(float decayRate = 0.99f, float minEpsilon = 0.01f);

    const Matrix<float> &q_table() const { return q_table_; }

  private:
    int nStates_;
    int nActions_;
    float alpha_;    // learning rate
    float gamma_;    // discount factor
    float epsilon_;  // exploration rate

    Matrix<float> q_table_;
    std::mt19937 rng_;
  };

}