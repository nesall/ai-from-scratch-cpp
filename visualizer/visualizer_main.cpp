#include "plotter.hpp"

extern void vis_linear_regression(SimplePlotter &plotter);
extern void vis_convolution(SimplePlotter &plotter);
extern void vis_optimizers(SimplePlotter &plotter);
extern void vis_lr_noiseinducing(SimplePlotter &plotter);
extern void vis_lr_others(SimplePlotter &plotter);

int main() {
  const unsigned W = 800;
  const unsigned H = 600;
  SimplePlotter plotter(W, H);

  //vis_linear_regression(plotter);
  //vis_convolution(plotter);
  //vis_optimizers(plotter);
  //vis_lr_noiseinducing(plotter);

  plotter.addTabBtn("Linear Regr.", [&] { vis_linear_regression(plotter); });
  plotter.addTabBtn("Convolutions", [&] { vis_convolution(plotter); });
  plotter.addTabBtn("Optimizers", [&] { vis_optimizers(plotter); });
  plotter.addTabBtn("NoiseInducingLR", [&] { vis_lr_noiseinducing(plotter); });
  plotter.addTabBtn("Cosine/Exp/Cyclical", [&] { vis_lr_others(plotter); });

  plotter.start();

  return 0;
}
