#include "plotter.hpp"

extern void vis_linear_regression(SimplePlotter &plotter);
extern void vis_convolution(SimplePlotter &plotter);
extern void vis_optimizers(SimplePlotter &plotter);
extern void vis_lr_noiseinducing(SimplePlotter &plotter);
extern void vis_lr_others(SimplePlotter &plotter);
extern void vis_ga(SimplePlotter &plotter);
extern void vis_kmeans(SimplePlotter &plotter);
extern void vis_qlearning(SimplePlotter &plotter);

int main() {
  SimplePlotter plotter(g_W, g_H);

  plotter.addTabBtn("Linear Regr.", [&] { vis_linear_regression(plotter); });
  plotter.addTabBtn("Convolutions", [&] { vis_convolution(plotter); }); //blurs a grayscale image
  plotter.addTabBtn("Optimizers", [&] { vis_optimizers(plotter); });
  plotter.addTabBtn("Noisy LR", [&] { vis_lr_noiseinducing(plotter); });
  plotter.addTabBtn("Other LR", [&] { vis_lr_others(plotter); });
  plotter.addTabBtn("GA", [&] { vis_ga(plotter); });
  plotter.addTabBtn("KMeans", [&] { vis_kmeans(plotter); });
  plotter.addTabBtn("Q-Learning", [&] { vis_qlearning(plotter); });

  plotter.start();

  return 0;
}
