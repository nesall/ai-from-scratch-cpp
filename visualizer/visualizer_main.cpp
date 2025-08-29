#include "plotter.hpp"

extern void vis_linear_regression(SimplePlotter &plotter);
extern void vis_convolution(SimplePlotter &plotter);
extern void vis_optimizers(SimplePlotter &plotter);
extern void vis_lr_noiseinducing(SimplePlotter &plotter);
extern void vis_lr_others(SimplePlotter &plotter);
extern void vis_ga(SimplePlotter &plotter);

int main() {
  const unsigned W = 800;
  const unsigned H = 600;
  SimplePlotter plotter(W, H);

  plotter.addTabBtn("Linear Regr.", [&] { vis_linear_regression(plotter); });
  plotter.addTabBtn("Convolutions", [&] { vis_convolution(plotter); });
  plotter.addTabBtn("Optimizers", [&] { vis_optimizers(plotter); });
  plotter.addTabBtn("Noisy LR", [&] { vis_lr_noiseinducing(plotter); });
  plotter.addTabBtn("Other LR", [&] { vis_lr_others(plotter); });
  plotter.addTabBtn("GA", [&] { vis_ga(plotter); });

  plotter.start();

  return 0;
}
