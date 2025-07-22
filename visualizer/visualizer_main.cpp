#include "plotter.hpp"

extern void vis_linear_regression(SimplePlotter &plotter);

int main() {
  SimplePlotter plotter(600, 400);

  vis_linear_regression(plotter);

  plotter.start();

  return 0;
}
