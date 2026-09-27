# dt-viz

Visualizations of the [Distribution Transformer](https://arxiv.org/abs/2502.02463) models
trained with [odin-ml](https://github.com/joekerenski/odin-ml), drawn with
[odin-ui-v2](https://github.com/joekerenski/odin-ui-v2).

## Radar

The M4 sensor-fusion model tracking targets it has never seen: the true track, each
rangefinder/bearing reading as a blip (true echo, clutter, or sensor failure), and the DT's
belief — a 4-component Gaussian mixture over position, as a density and one ellipse per
component — updated once per reading and pushed through the dynamics in between. A
5000-particle filter runs alongside; the timeline scores both filters at every step
(−log density at the true state; lower is better).

```sh
make model    # once: trains the M4 model with the pinned odin-ml (~40 min on an M-series GPU)
make run      # clones odin-ml + odin-ui-v2 into deps/ (pinned), builds, runs
make run MODEL=../odin-ml/models/fusion_dt4.safetensors   # use a model trained elsewhere
make run ML_DIR=../odin-ml UI_DIR=../odin-ui-v2           # develop against local checkouts
```

odin-ml keeps trained weights out of git, so the model lives in `models/` here (ignored).

Space play/pause · ←/→ step · N new track · F fullscreen · drag the timeline to scrub.

Both libraries are imported as Odin collections (`odinml:ml`, `odinml:dt`, `odinui:ui`),
pinned in the Makefile. Inter font: SIL Open Font License (assets/Inter-LICENSE.txt).
