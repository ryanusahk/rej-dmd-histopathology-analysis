# Manuscript Code Map

This file maps manuscript analysis components to the deposited code.

| Manuscript component | Relevant code | Notes |
|---|---|---|
| AAV co-transduction / multi-vector dose modeling | `analysis/01_aav_poisson_dose_model.ipynb` | Uses Poisson modeling to relate dose, estimated MOI, and single / dual / triple vector co-expression. |
| Cellpose segmentation and early image processing | `analysis/02_cellpose_segmentation.ipynb` | Contains segmentation workflow and related image-processing exploration. |
| Whole-muscle myofiber quantification | `analysis/03_histology_quantification_and_figures.ipynb` | Contains myofiber measurements, central nuclei, Feret diameter / hypertrophy, dystrophin intensity, classifier application, and plotting provenance. |
| Dystrophin positivity classifier / active learning | `scripts/active_learning.py`, `scripts/active_training_quant.py`, `scripts/active_learning_eval.py`, `scripts/learning_curve_estimator.py`, `analysis/03_histology_quantification_and_figures.ipynb` | Utility scripts and notebook cells supporting active-learning classifier development and validation. |
| Myofiber atypicality classifier | `analysis/03_histology_quantification_and_figures.ipynb` | Includes classifier development and application for typical / atypical myofiber morphology. |
| Spatial neighbor analysis | `analysis/03_histology_quantification_and_figures.ipynb` | Includes Delaunay-neighbor / spatial graph analysis of dystrophin-positive and dystrophin-negative fibers. |
| Statistical power and clustered analysis framework | `analysis/04_power_analysis.ipynb` | Manuscript-relevant clustered GEE / power framework using an exchangeable correlation structure. |

## Statistical Interpretation

For manuscript-relevant statistical interpretation of clustered myofiber data, use `analysis/04_power_analysis.ipynb`. Some statistical cells in `analysis/03_histology_quantification_and_figures.ipynb` are legacy exploratory outputs retained as analysis provenance.
