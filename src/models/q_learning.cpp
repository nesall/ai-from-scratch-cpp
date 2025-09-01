#include "models/q_learning.hpp"
#include <cassert>

utils::Gridworld::Gridworld(int width, int height, 
  std::pair<int, int> start, std::pair<int, int> goal, 
  const std::vector<std::pair<int, int>> &traps)
  : width_(width), height_(height), start_(start), goal_(goal), traps_(traps), state_(start)
{
}

void utils::Gridworld::reset()
{
  state_ = start_;
}

bool utils::Gridworld::isTerminal(const std::pair<int, int> &state) const
{
  if (state == goal_) return true;
  for (const auto &trap : traps_) {
    if (state == trap) return true;
  }
  return false;
}

std::tuple<std::pair<int, int>, float, bool> utils::Gridworld::step(Action action)
{
  if (isTerminal(state_)) {
    return std::make_tuple(state_, 0.0f, true);
  }
  std::pair<int, int> next_state = state_;
  switch (action) {
  case UP:
    if (state_.second > 0) next_state.second -= 1;
    break;
  case DOWN:
    if (state_.second < height_ - 1) next_state.second += 1;
    break;
  case LEFT:
    if (state_.first > 0) next_state.first -= 1;
    break;
  case RIGHT:
    if (state_.first < width_ - 1) next_state.first += 1;
    break;
  default:
    assert(!"Invalid action");
    break;
  }
  state_ = next_state;
  float reward = -0.1f; // small negative reward for each step
  bool done = false;
  if (state_ == goal_) {
    reward = 1.0f; // reward for reaching goal
    done = true;
  } else {
    for (const auto &trap : traps_) {
      if (state_ == trap) {
        reward = -1.0f; // penalty for falling into trap
        done = true;
        break;
      }
    }
  }
  return std::make_tuple(state_, reward, done);
}

int utils::Gridworld::state2index(const std::pair<int, int> &state) const
{
  assert(!(state.first < 0 || state.first >= width_ || state.second < 0 || state.second >= height_));
  return state.second * width_ + state.first;
}



models::Q_LearningAgent::Q_LearningAgent(int nstates, int nactions, float alpha, float gamma, float epsilon)
  : nStates_(nstates), nActions_(nactions), alpha_(alpha), gamma_(gamma), epsilon_(epsilon),
  q_table_(nstates, nactions), rng_(std::random_device{}())
{
  assert(nstates > 0 && nactions > 0);
}

int models::Q_LearningAgent::selectAction(int state)
{
  assert(state >= 0 && state < nStates_);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  if (dist(rng_) < epsilon_) {
    // Explore: random action
    std::uniform_int_distribution<int> action_dist(0, nActions_ - 1);
    return action_dist(rng_);
  } else {
    // Exploit: best known action
    return bestKnownAction(state);
  }
  return 0;
}

int models::Q_LearningAgent::bestKnownAction(int state) const
{
  int bestAction = 0;
  float maxQ = q_table_.at(state, 0);
  for (int a = 1; a < nActions_; ++a) {
    float q = q_table_.at(state, a);
    if (q > maxQ) {
      maxQ = q;
      bestAction = a;
    }
  }
  return bestAction;
}

void models::Q_LearningAgent::update(int state, int action, float reward, int nextState, bool done)
{
  // Q(s,a) = Q(s,a) + alpha * (reward + gamma * max_a' Q(s',a') - Q(s,a))
  float q_sa = q_table_.at(state, action);
  float max_q_next = 0.0f;
  if (!done) {
    max_q_next = q_table_.at(nextState, 0);
    for (int a = 1; a < nActions_; ++a) {
      max_q_next = std::max(max_q_next, q_table_.at(nextState, a));
    }
  }
  q_table_.at(state, action) = q_sa + alpha_ * (reward + gamma_ * max_q_next - q_sa);
}

void models::Q_LearningAgent::decayEpsilon(float decayRate, float minEpsilon)
{
  epsilon_ = std::max(minEpsilon, epsilon_ * decayRate);
}
