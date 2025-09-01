#include "plotter.hpp"
#include "models/q_learning.hpp"
#include <random>
#include <algorithm>
#include <map>


std::string actToStr(utils::Gridworld::Action a) {
  switch (a) {
  case utils::Gridworld::Action::UP: return /*"UP"*/"DOWN"; // because y axis is drawn inverted
  case utils::Gridworld::Action::DOWN: return /*"DOWN"*/"UP"; // because y axis is drawn inverted
  case utils::Gridworld::Action::LEFT: return "LEFT";
  case utils::Gridworld::Action::RIGHT: return "RIGHT";
  default: return "UNKNOWN";
  }
}

void draw(SimplePlotter &plotter, const utils::Gridworld &env, const models::Q_LearningAgent &agent) {
  plotter.clearShapes();
  auto cols = env.cols();
  auto rows = env.rows();
  auto goal = env.goal();
  auto state = env.state();
  auto traps = env.traps();
  const auto &qtab = agent.q_table();
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      sf::Color clr;
      if (std::pair{ x, y } == goal) {
        clr = sf::Color::Green;
      } else if (std::find(traps.cbegin(), traps.cend(), std::pair{ x, y }) != traps.cend()) {
        clr = sf::Color(255, 165, 200);
      } else {
        clr = sf::Color(200, 200, 200);
      }
      plotter.addCircle(10.f, { plotter.toScreenX(x + 0.5f), plotter.toScreenY(y + 0.5f) }, clr);
      auto stateIndex = env.state2index(std::pair{ x, y });
      auto row = qtab[stateIndex];
      if (0 != *std::max_element(row.begin(), row.end())) {
        auto action = agent.bestKnownAction(stateIndex);
        auto str = actToStr(static_cast<utils::Gridworld::Action>(action));
        plotter.addText({ plotter.toScreenX(x + 0.2f), plotter.toScreenY(y + 0.3f) }, str, sf::Color::Black);
      }
    }
  }
  plotter.addCircle(10.f, { plotter.toScreenX(state.first + 0.5f), plotter.toScreenY(state.second + 0.5f) }, sf::Color::Blue);

  sf::sleep(sf::milliseconds(0));
  plotter.drawNextFrame();
  sf::sleep(sf::milliseconds(33));
}

void vis_qlearning(SimplePlotter &plotter) {
  std::cout << "vis_qlearning\n";
  plotter.setWindowSize(g_W, g_H);
  plotter.clearShapes();
  plotter.setDrawAxes(false);

  const int rows = 5, cols = 5;
  const sf::Vector2i goal(4, 4);
  std::mt19937 rng(std::random_device{}());

  plotter.setDataBounds(0, cols, 0, rows);

  utils::Gridworld env(cols, rows, { 0, 0 }, { 4, 4 }, { {1, 1}, {2, 2}, {3, 3} });

  const float alpha = 0.1f, gamma = 0.9f, epsilon = 0.2f;
  const int episodes = 100, maxSteps = 30;
  models::Q_LearningAgent agent(env.nofStates(), env.nofActions(), alpha, gamma, epsilon);

  for (int ep = 0; ep < episodes; ++ep) {
    env.reset();
    auto state = env.state();
    int stateIndex = env.state2index(state);
    float totalReward = 0.0f;
    bool done = false;
    while (!done) {
      int a = agent.selectAction(stateIndex);
      auto [nextState, reward, finished] = env.step(static_cast<utils::Gridworld::Action>(a));
      int s_next = env.state2index(nextState);
      agent.update(stateIndex, a, reward, s_next, finished);
      draw(plotter, env, agent);
      stateIndex = s_next;
      totalReward += reward;
      done = finished;
    }
    agent.decayEpsilon(0.995f);
  }

  draw(plotter, env, agent);

}
