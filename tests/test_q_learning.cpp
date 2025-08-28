#include "models/q_learning.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <chrono>
#include <thread>

// Render the grid visually in ASCII
void render(const utils::Gridworld &env, const std::pair<int, int> &agent_pos) {
  auto traps = env.traps();
  for (int r = 0; r < env.rows(); ++r) {
    for (int c = 0; c < env.cols(); ++c) {
      if (agent_pos.first == r && agent_pos.second == c) {
        std::cout << "A "; // agent
      } else if (env.goal() == std::make_pair(r, c)) {
        std::cout << "G "; // goal
      } else if (std::count(traps.cbegin(), traps.cend(), std::make_pair(r, c))) {
        std::cout << "X "; // trap
      } else if (env.start() == std::make_pair(r, c)) {
        std::cout << "S "; // start
      } else {
        std::cout << ". ";
      }
    }
    std::cout << "\n";
  }
  std::cout << "-----------------\n";
}

void test_q_learning() {
  using Action = utils::Gridworld::Action;

  // Gridworld: 4x4, start (0,0), goal (3,3), traps at (1,1) and (2,2)
  utils::Gridworld env(4, 4, { 0,0 }, { 3,3 }, { {1,1}, {2,2} });
  models::Q_LearningAgent agent(env.nofStates(), env.nofActions(),
    0.1f,   // alpha
    0.99f,  // gamma
    0.2f);  // epsilon

  const int episodes = 200;
  for (int ep = 0; ep < episodes; ++ep) {
    env.reset();
    auto state = env.getState();
    int s = env.state2index(state);

    float totalReward = 0.0f;
    bool done = false;

    while (!done) {
      int a = agent.selectAction(s);
      auto [next_state, reward, finished] = env.step(static_cast<Action>(a));
      int s_next = env.state2index(next_state);

      agent.update(s, a, reward, s_next, finished);

      s = s_next;
      totalReward += reward;
      done = finished;
    }

    agent.decayEpsilon(0.995f);

    std::cout << "Episode " << ep + 1 << " total reward: " << totalReward << "\n";
  }

  // Print learned Q-table
  std::cout << "\nLearned Q-table:\n";
  const auto &Q = agent.q_table();
  for (size_t s = 0; s < Q.rows(); ++s) {
    std::cout << "State " << s << ": ";
    for (float qval : Q[s]) {
      std::cout << std::setw(8) << std::fixed << std::setprecision(3) << qval << " ";
    }
    std::cout << "\n";
  }

  // Test the trained agent visually
  env.reset();
  auto state = env.getState();
  int s = env.state2index(state);

  bool done = false;
  while (!done) {
    render(env, env.getState());
    int a = agent.selectAction(s);
    auto [next_state, reward, finished] = env.step(static_cast<Action>(a));
    s = env.state2index(next_state);
    done = finished;
    std::this_thread::sleep_for(std::chrono::milliseconds(1000)); // pause to animate
  }

  render(env, env.getState());
  std::cout << "Final reward: " << (env.getState() == env.goal() ? "Reached Goal!" : "Fell in Trap!") << "\n";

}
