# REJ-DMD Histopathology Analysis Code

This repository contains analysis code associated with the manuscript on RNA trans-splicing treatment for Duchenne muscular dystrophy. The deposited workflows support two computational components of the study: quantitative AAV co-transduction dose modeling and high-throughput computational histopathology of dystrophic muscle.

The histopathology workflow combines neural-network-based myofiber segmentation, channel-specific computer-vision feature engineering, human-in-the-loop machine learning, spatial tissue-architecture analysis, and cluster-aware statistical modeling. Together, these analyses quantify treatment-associated changes across whole muscle cross-sections at single-myofiber resolution.

This repository is intended as an analysis-code deposit. Full re-execution requires large raw microscopy images and intermediate files that are not included here.

## Workflow Overview

The deposited analysis code covers:

- Poisson modeling of AAV transduction and multi-vector co-expression to estimate the dose relationships required for single-, dual-, and triple-vector expression.
- Neural-network-assisted segmentation of immunofluorescence muscle images using Cellpose to identify individual myofiber boundaries from membrane staining.
- Myofiber-level computer-vision feature extraction from multiplexed fluorescence channels, including Feret diameter, dystrophin intensity, membrane-associated signal, cytoplasmic nuclear signal, central nuclei features, and nuclear infiltration / tissue damage-associated features.
- Human-in-the-loop active-learning workflows for training and refining random forest classifiers to distinguish dystrophin-positive and dystrophin-negative myofibers.
- Machine-learning analysis of myofiber morphology, including typical / atypical phenotype classification based on structural image features independent of the dystrophin signal.
- Spatial network analysis of tissue architecture using Delaunay triangulation to model neighboring myofiber relationships and evaluate local effects of dystrophin-positive fibers.
- Cluster-aware statistical modeling and power analysis for nested myofiber data, including generalized estimating equation frameworks that account for myofibers clustered within muscle samples.

## Contents

- `analysis/01_aav_poisson_dose_model.ipynb`: Poisson modeling of AAV dose, transduction, and multi-vector co-expression.
- `analysis/02_cellpose_segmentation.ipynb`: Cellpose-based segmentation workflow and early image-processing analysis.
- `analysis/03_histology_quantification_and_figures.ipynb`: Myofiber feature extraction, histology quantification, classifier application, plotting, and spatial analysis provenance.
- `analysis/04_power_analysis.ipynb`: Manuscript-relevant clustered GEE / power analysis framework.
- `scripts/`: Active-learning and classifier-evaluation utilities used for dystrophin-positive / dystrophin-negative myofiber classification.

## Notebook Cleanup

The notebooks in this deposit are cleaned copies prepared for readability. The analysis code is preserved; notebook metadata and embedded outputs were cleaned for readability.

## Statistical Note

Some statistical cells in `analysis/03_histology_quantification_and_figures.ipynb` are retained as legacy exploratory analysis and may use older assumptions. Manuscript-relevant clustered GEE / power analyses are documented in `analysis/04_power_analysis.ipynb`, which uses an exchangeable correlation structure for myofibers nested within muscle samples.

## Scope

This deposit does not include analysis code for every assay in the manuscript. RNA-seq, physiology, behavior, Western blot, and molecular assay analysis code are outside the scope of this repository.

## Data Dependencies

See `DATA_AVAILABILITY.md` for expected input files and directories required for full re-execution.
