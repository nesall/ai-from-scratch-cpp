#include "plotter.hpp"
#include "models/optimizers.hpp"
#include "datasynth.hpp"


// Analysis of optimizer behavior on this loss landscape

class OptimizerAnalysis {
public:
  // Your loss function: (theta-5)^2 + 2*sin(3*theta) + noise
  // Critical point at Theta ~= 5.683 where gradient ~= 0

  static void analyze_success_factors() {
    std::cout << "\n=== WHY OPTIMIZERS SUCCEEDED/FAILED ===\n\n";

    std::cout << "1. SGD with Decay (SUCCESS):\n";
    std::cout << "   . High initial LR (0.195) provided escape velocity\n";
    std::cout << "   . Decay schedule refined search near optimum\n";
    std::cout << "   . Simple momentum-free updates avoided oscillation\n";
    std::cout << "   . Found stable critical point theta ~= 5.683\n\n";

    std::cout << "2. Momentum (SUCCESS):\n";
    std::cout << "   . Moderate LR (0.05) with beta=0.9 momentum\n";
    std::cout << "   . Momentum helped escape early local minima\n";
    std::cout << "   . Converged to same critical point theta ~= 5.683\n";
    std::cout << "   . Velocity dampening provided stability\n\n";

    std::cout << "3. Adam (SUCCESS after tuning):\n";
    std::cout << "   . Required high LR (0.4) to escape local trap\n";
    std::cout << "   . Beta2=0.999 provided more stable second moments\n";
    std::cout << "   . Eventually found same critical point theta ~= 5.683\n";
    std::cout << "   . Adaptive nature helped once in right basin\n\n";

    std::cout << "4. RMSProp (FAILURE):\n";
    std::cout << "   . Trapped in oscillatory region around theta ~= 3.84\n";
    std::cout << "   . Adaptive LR creates vicious cycle:\n";
    std::cout << "     - Large gradients -> smaller effective LR\n";
    std::cout << "     - Smaller LR -> can't escape oscillation\n";
    std::cout << "     - Continues oscillating -> maintains large gradients\n";
    std::cout << "   . Even lr=0.45 insufficient to break cycle\n";
  }

  static void loss_landscape_insights() {
    std::cout << "\n=== LOSS LANDSCAPE INSIGHTS ===\n\n";

    std::cout << "The function f(theta) = (theta-5)^2 + 2*sin(3theta) has:\n";
    std::cout << ". Quadratic bowl centered at theta=5\n";
    std::cout << ". Sinusoidal oscillations with period 2pi/3 ~= 2.09\n";
    std::cout << ". Multiple local minima and maxima\n\n";

    std::cout << "Critical point theta ~= 5.683:\n";
    std::cout << ". Located 0.683 units right of parabola minimum\n";
    std::cout << ". Where quadratic slope balances sinusoidal slope\n";
    std::cout << ". 2*(theta-5) + 6*cos(3theta) = 0\n";
    std::cout << ". Very stable - gradients increase linearly away\n\n";

    std::cout << "RMSProp trap theta ~= 3.84:\n";
    std::cout << ". Not a true critical point (grad != 0)\n";
    std::cout << ". Located in saddle/unstable region\n";
    std::cout << ". Creates oscillatory dynamics for adaptive methods\n";
  }

  static void practical_lessons() {
    std::cout << "\n=== PRACTICAL LESSONS ===\n\n";

    std::cout << "1. No universal optimizer - landscape matters!\n";
    std::cout << "2. Learning rate is often more important than algorithm choice\n";
    std::cout << "3. Learning rate schedules can make simple methods competitive\n";
    std::cout << "4. Adaptive methods can get trapped by their own adaptation\n";
    std::cout << "5. Multiple random restarts help find better solutions\n";
    std::cout << "6. Always check if gradients -> 0 to verify convergence\n";
    std::cout << "7. Visualize loss landscape when possible\n";

    std::cout << "\n=== RECOMMENDATIONS FOR COMPLEX LOSS LANDSCAPES ===\n\n";
    std::cout << "1. Multi-start optimization with different seeds\n";
    std::cout << "2. Learning rate scheduling (warmup, cosine decay, etc.)\n";
    std::cout << "3. Gradient clipping for stability\n";
    std::cout << "4. Hybrid approaches (start with SGD, finish with Adam)\n";
  }
};


class SGDWithDecay {
private:
  float initial_lr_;
  float decay_rate_; // [0, 1)
  int step_count_ = 0;

public:
  SGDWithDecay(float lr, float decay = 0.999f) : initial_lr_(lr), decay_rate_(decay) {}

  void update(std::span<float> params, std::span<const float> grads, int context = 0) {
    step_count_++;
    float current_lr = initial_lr_ * std::pow(decay_rate_, step_count_);
    for (size_t i = 0; i < params.size(); ++i) {
      params[i] -= current_lr * grads[i];
    }
  }
};

// Global RNG for noise
std::vector<float> generate_noise(int steps) {
  std::mt19937 rng(42);
  std::normal_distribution<float> noise_dist(0.0f, 0.1f);
  std::vector<float> noise(steps);
  for (int i = 0; i < steps; i++) {
    noise[i] = noise_dist(rng);
  }
  return noise;
}

float loss(float theta, float noise) {
  return (theta - 5.0f) * (theta - 5.0f) + 2.0f * std::sin(3.0f * theta) + noise;
}

// Gradient of the loss (derivative wrt theta, ignoring noise for stability)
float grad(float theta) {
  return 2.0f * (theta - 5.0f) + 6.0f * std::cos(3.0f * theta);
}

std::vector<float> run_optimizer(const std::string &name, size_t steps, auto &optimizer, const std::vector<float> &noise) {
  std::vector<float> params = { -5.0f }; // start far from optimum
  std::vector<float> losses;
  losses.reserve(steps);
  for (int i = 0; i < steps; ++i) {
    std::vector<float> grads = { grad(params[0]) };
    optimizer.update(params, grads);
    losses.push_back(loss(params[0], noise[i]));

    if (steps - 5 < i) {
      std::cout << name << " grad " << grads[0] << "; param " << params[0] << "; loss " << losses[i] << "\n";
      
    }
  }
  return losses;
}

void vis_optimizers(SimplePlotter &plotter) {
  const int steps = 1000;
  optimizers::SGD sgd(0.195f);
  sgd.reset_scheduler(std::make_unique<optimizers::StepDecayLR>(std::vector<int>{100, 500}));
  optimizers::Momentum mom(0.05f, 0.9f);
  optimizers::RMSProp rms(0.45f, 0.9f);
  rms.reset_scheduler(std::make_unique<optimizers::WarmupCosineDecayLR>(100, steps));
  optimizers::Adam adm(0.4f, 0.9f, 0.999f);

  auto noise = generate_noise(steps);

  auto loss_sgd = run_optimizer("SGD", steps, sgd, noise);
  auto loss_mom = run_optimizer("Momentum", steps, mom, noise);
  auto loss_rms = run_optimizer("RMSProp", steps, rms, noise);
  auto loss_adm = run_optimizer("Adam", steps, adm, noise);

  auto [minsgd, maxsgd] = std::minmax_element(loss_sgd.cbegin(), loss_sgd.cend());
  auto [minmom, maxmom] = std::minmax_element(loss_mom.cbegin(), loss_mom.cend());
  auto [minrms, maxrms] = std::minmax_element(loss_rms.cbegin(), loss_rms.cend());
  auto [minadm, maxadm] = std::minmax_element(loss_adm.cbegin(), loss_adm.cend());
  std::vector<float> mm{ *minsgd, *maxsgd, *minmom, *maxmom, *minrms, *maxrms, *minadm, *maxadm };
  auto [minval, maxval] = std::minmax_element(mm.cbegin(), mm.cend());
  
  plotter.setDataBounds(0.f, static_cast<float>(steps), *minval, *maxval);
  for (size_t i = 0; i < steps; ++i) {
    plotter.addCircle(2.f, sf::Vector2f(plotter.toScreenX(static_cast<float>(i)), plotter.toScreenY(loss_sgd[i])), sf::Color::Blue);
  }
  for (size_t i = 0; i < loss_mom.size(); ++i) {
    plotter.addCircle(2.f, sf::Vector2f(plotter.toScreenX(static_cast<float>(i)), plotter.toScreenY(loss_mom[i])), sf::Color::Red);
  }
  for (size_t i = 0; i < loss_rms.size(); ++i) {
    plotter.addCircle(2.f, sf::Vector2f(plotter.toScreenX(static_cast<float>(i)), plotter.toScreenY(loss_rms[i])), sf::Color::Green);
  }
  for (size_t i = 0; i < loss_adm.size(); ++i) {
    plotter.addCircle(2.f, sf::Vector2f(plotter.toScreenX(static_cast<float>(i)), plotter.toScreenY(loss_adm[i])), sf::Color::Magenta);
  }
  
  // Add legend name/color
  plotter.addText(sf::Vector2f(plotter.toScreenX(steps - 20), plotter.toScreenY(25.f)), "SGD", sf::Color::Blue);
  plotter.addText(sf::Vector2f(plotter.toScreenX(steps - 20), plotter.toScreenY(22.f)), "Momentum", sf::Color::Red);
  plotter.addText(sf::Vector2f(plotter.toScreenX(steps - 20), plotter.toScreenY(19.f)), "RMSProp", sf::Color::Green);
  plotter.addText(sf::Vector2f(plotter.toScreenX(steps - 20), plotter.toScreenY(16.f)), "Adam", sf::Color::Magenta);
  

  OptimizerAnalysis::analyze_success_factors();
  OptimizerAnalysis::loss_landscape_insights();
  OptimizerAnalysis::practical_lessons();
}
