# Market Volatility Prediction

This project studies market volatility prediction using publicly available
financial data, with a focus on the statistical, geometric, and topological
structure of financial time series.

The project is conceptually based on an earlier internship project on
volatility prediction. Since the original project used proprietary financial
features, this repository reconstructs the core workflow using publicly
available SPY data and then develops several research directions beyond the
original project.

The project is organized into four stages:

1. **Stage 0 — Reproduce the Internship**
2. **Stage 1 — One-Dimensional Convolutional Neural Network**
3. **Stage 2 — Data Augmentation and Geometric Deep Learning**
4. **Stage 3 — Renormalization-Group Flow**

The stages are designed to progress from a reproducible machine-learning
baseline toward increasingly mathematical and structure-aware approaches.

---

## Dataset

The project uses daily SPY (SPDR S&P 500 ETF Trust) data from January 2010
to the present.

The long historical period allows the analysis to include multiple market
and volatility regimes.

The data are downloaded using `yfinance`.

Raw and processed datasets are not committed to the repository.

---

## Project Structure

```text
market-volatility/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── figures/
├── models/
├── notebooks/
├── outputs/
├── src/
├── .gitignore
├── README.md
└── requirements.txt
```
## Research Roadmap
### Stage 0 — Baseline Volatility Prediction

Stage 0 establishes a baseline for predicting whether future five-day realized volatility will substantially exceed current volatility, using daily SPY data from January 2010 onward.

The target is constructed from the ratio of future five-day realized volatility to backward-looking 20-day volatility. A log-ratio threshold of 0.2 defines the positive class.

The workflow includes:

- Data acquisition and cleaning
- Exploratory data analysis and volatility characterization
- Topological data analysis of sliding-window return embeddings
- Feature engineering using conventional volatility statistics and TDA-derived features
- Chronological training, validation, and test splits
- Model training, hyperparameter tuning, and threshold selection

Four models were evaluated on the same chronological test period (2023–2026):

| Model | Accuracy | Balanced Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.5887 | 0.6765 | 0.3127 | 0.8263 | **0.4538** | **0.7557** | **0.4217** |
| Neural Network | 0.6605 | 0.6673 | 0.3395 | 0.6789 | 0.4526 | 0.7249 | 0.3813 |
| XGBoost | 0.5713 | 0.6636 | 0.3023 | 0.8211 | 0.4419 | 0.7359 | 0.3901 |
| Logistic Regression | 0.7998 | 0.5547 | 0.5652 | 0.1368 | 0.2203 | 0.7353 | 0.4099 |

Random Forest achieved the highest test F1, ROC-AUC, and PR-AUC, while the neural network achieved a very similar F1 score. Logistic Regression had the highest accuracy but substantially lower recall.

These results provide a baseline for the subsequent recurrent neural network experiments. Although the feature set includes TDA-derived descriptors, this comparison does not isolate their incremental contribution; that would require an ablation study comparing models with and without TDA features.

**Reproducibility:** The notebooks document data preparation, feature construction, model selection, evaluation, and the selected classification thresholds. The `src/` directory contains reusable project utilities. Locally saved model artifacts can be used to reproduce predictions without retraining, subject to compatible software versions.

### Stage 1 — Recurrent Neural Network (RNN)

Stage 1 investigates recurrent neural networks for five-day-ahead volatility prediction using sequential SPY data. Building on the Stage-0 baseline, this stage explores whether recurrent architectures can better capture temporal dependencies in financial returns and volatility.

The experiment will evaluate RNN-based predictions against the Stage-0 models using the same chronological evaluation framework. The methodology and conclusions will be updated after the experiments are complete.

### Stage 2 - One-Dimensional Convolutional Neural Network

Stage 1 investigates whether a one-dimensional convolutional neural network
can learn useful local temporal structure directly from financial time series.

Rather than relying entirely on manually engineered features, the model will
operate on sequences of returns or related financial quantities.

Questions of interest include:

Can a 1D CNN learn useful local patterns in financial time series?
How does it compare with models based on engineered features?
What role does temporal locality play in volatility prediction?
Can learned representations reveal structure not captured by conventional
feature engineering?

This stage provides a transition from conventional machine learning toward
geometric and structure-aware learning.

### Stage 3 - Data Augmentation and Geometric Deep Learning

The central motivation for this stage comes from the internship project. The
financial dataset was relatively small and low-to-moderate dimensional, and
deep-learning models did not perform as well as the leading tree-based model,
XGBoost.

This motivates the question:

> Can symmetry of the prediction function, together with the topology and
> distribution of the data, be used to construct useful data augmentation?

The goal is not simply to generate additional observations. An augmentation
should be scientifically justified from three complementary perspectives:

1. **Symmetry of the prediction function**

   Identify transformations under which the prediction function is invariant
   or transforms in a known way.

2. **Distribution of the data**

   Determine whether the transformed observations remain compatible with the
   statistical distribution of the original financial data.

3. **Topology and geometry of the data**

   Determine whether the transformation preserves or appropriately modifies
   the geometric and topological structure of the observations.

These considerations lead to the broader question of whether structure-aware
data augmentation can make deep-learning models more effective in the
low-to-moderate data regime encountered in financial time-series prediction.

Topological Data Analysis provides tools for studying the effect of candidate
augmentations on the structure of the data, while Geometric Deep Learning
provides a framework for thinking about symmetry, invariance, and
equivariance.

The objective is therefore to test whether carefully constructed,
mathematically motivated augmentations can improve predictive performance
relative to training on the original dataset alone.

### Stage 4 — Renormalization-Group Flow

Stage 3 explores renormalization-group ideas for financial time-series
dynamics.

The central question is how the statistical, geometric, and topological
structure of the system changes under transformations of scale.

Possible directions include:

- temporal coarse-graining,
- multiscale representations,
- evolution of return distributions under coarse-graining,
- volatility across scales,
- evolution of topological quantities across scales,
- and the relationship between coarse-grained representations and
prediction.

This stage is exploratory and will be developed after the earlier stages
have established the relevant empirical and geometric structure.

## Mathematical Themes

The project brings together:

- time-series analysis,
- volatility modeling,
- machine learning,
- convolutional neural networks,
- geometric deep learning,
- persistent homology,
- topological data analysis,
- symmetry and invariance,
- data augmentation,
- stochastic processes,
- and renormalization-group methods.

A guiding principle is to study statistical, geometric, and topological
structure as complementary aspects of the same learning problem.

## Topological Data Analysis

The TDA component uses:

- delay-coordinate embeddings,
- sliding windows,
- persistent homology,
- persistence diagrams,
- persistence landscapes,
- and landscape norms.

The implementation uses:

- ripser
- persim

The mathematical and computational background for TDA is developed further
in the accompanying repository:

https://github.com/vaishampayan1729/topological-data-analysis

The financial TDA analysis is motivated in part by:

>M. Gidea and Y. Katz, "Topological Data Analysis of Financial Time Series:
>Landscapes of Crashes," Physica A: Statistical Mechanics and its
>Applications, vol. 491, pp. 820--834, 2018.

The present project does not attempt to reproduce that study exactly.
Instead, it adapts the general idea of extracting topological information
from financial time series to a single publicly available asset.

Environment

The project uses a dedicated Conda environment.

Install the required packages with:

>pip install -r requirements.txt

## Status

Stage 0 is currently in progress.

Completed:

- data loading,
- data cleaning,
- exploratory data analysis,
- topological data analysis.

The next step is to construct the feature set for the Stage-0
volatility-prediction baseline.