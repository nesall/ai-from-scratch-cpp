#include "plotter.hpp"

extern void vis_linear_regression(SimplePlotter &plotter);
extern void vis_convolution(SimplePlotter &plotter);

int main() {
  SimplePlotter plotter(600, 400);

  //vis_linear_regression(plotter);
  vis_convolution(plotter);

  plotter.start();

  return 0;
}
