
# Market Volatility Prediction

This project studies market volatility prediction using publicly available financial data, with an emphasis on the statistical, geometric, and topological structure of financial time series.

The project is conceptually based on an earlier internship project on volatility prediction. Since the original project used proprietary financial features, this repository reconstructs the core workflow using publicly available SPY data and explores several research directions beyond the original project.

The project is organized into five stages:

1. **Stage 0 — Baseline Volatility Prediction**
2. **Stage 1 — Recurrent Neural Networks**
3. **Stage 2 — One-Dimensional Convolutional Neural Networks**
4. **Stage 3 — Data Augmentation and Geometric Deep Learning**
5. **Stage 4 — Renormalization-Group Flow**

The stages progress from reproducible machine-learning baselines toward increasingly mathematical and structure-aware approaches.

---

## Dataset

The project uses daily SPY (SPDR S&P 500 ETF Trust) data beginning in January 2010.

Data are downloaded using `yfinance`.

The historical period includes multiple market and volatility regimes, allowing us to investigate the behavior of predictive models under changing market conditions.

Raw and processed datasets are not committed to the repository.

### Prediction Target

The objective is to predict whether future five-day realized volatility substantially exceeds current volatility.

The binary target is defined as

$$
y_t =
\mathbb{1}\left\{
\log\left(
\frac{RV_{t,t+5}}{Vol_{20,t}}
\right)>0.2
\right\},
$$

where:

- $RV_{t,t+5}$ denotes realized volatility over the next five trading days.
- $Vol_{20,t}$ denotes backward-looking 20-day volatility.

Thus, the positive class represents a sufficiently large increase in future realized volatility relative to current volatility.

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
│   └── gru_final/
│       ├── gru_seed_42.keras
│       ├── gru_seed_123.keras
│       ├── gru_seed_2026.keras
│       ├── config.json
│       ├── inference_config.json
│       ├── scaler.joblib
│       └── test_results.csv
├── notebooks/
│   ├── ...
│   ├── 10_sequence_models.ipynb
│   └── 11_sequence_models_rnn.ipynb
├── outputs/
├── src/
├── .gitignore
├── README.md
└── requirements.txt
```

The `notebooks/` directory contains the experimental and educational notebooks, while `src/` is intended for reusable project utilities.

The `models/` directory stores trained model artifacts and preprocessing objects. Large model files may be excluded from Git version control.

### Notebook Organization

The notebooks are organized approximately according to the research stages.

| Notebook | Description |
|---|---|
| Stage-0 notebooks | Data acquisition, cleaning, exploratory analysis, TDA, feature engineering, and classical machine-learning models |
| `10_sequence_models.ipynb` | Classical time-series models, including autoregressive and moving-average approaches |
| `11_sequence_models_rnn.ipynb` | Simple RNN, LSTM, and GRU architectures; hyperparameter experiments; model comparisons; final GRU evaluation |

The notebook numbering follows the development of the project. Individual notebooks are designed to be reasonably self-contained while sharing the same underlying prediction problem.

### A Note on the Notebook Style

**Beginning with Stage 1, the notebooks follow a slightly more tutorial-oriented approach.**

In addition to presenting experiments and results, they aim to explain the mathematical and computational ideas underlying the models.

In particular, the notebooks may include:

- Mathematical definitions and theoretical motivation.
- Explanations of model architectures and their components.
- Examples illustrating important concepts.
- Step-by-step implementations using Python and deep-learning libraries.
- Controlled experiments investigating architectural and hyperparameter choices.
- Discussion and interpretation of empirical results.

The objective is twofold: to conduct reproducible research experiments and to develop a deeper understanding of the methods being implemented.

Consequently, some notebooks contain more exposition than is strictly necessary for a conventional experimental pipeline.

They are intended to serve both as research records and as educational resources.

---

## Research Roadmap

### Stage 0 — Baseline Volatility Prediction

**Status: Completed**

Stage 0 establishes a baseline for predicting five-day-ahead volatility increases using daily SPY data.

The workflow includes:

- Data acquisition and cleaning.
- Exploratory data analysis and volatility characterization.
- Topological data analysis of sliding-window return embeddings.
- Feature engineering using conventional volatility statistics and TDA-derived features.
- Chronological training, validation, and test splits.
- Model training, hyperparameter tuning, and threshold selection.

Four models were evaluated on the same chronological test period (2023–2026):

| Model | Accuracy | Balanced Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.5887 | 0.6765 | 0.3127 | 0.8263 | **0.4538** | **0.7557** | **0.4217** |
| Neural Network | 0.6605 | 0.6673 | 0.3395 | 0.6789 | 0.4526 | 0.7249 | 0.3813 |
| XGBoost | 0.5713 | 0.6636 | 0.3023 | 0.8211 | 0.4419 | 0.7359 | 0.3901 |
| Logistic Regression | 0.7998 | 0.5547 | 0.5652 | 0.1368 | 0.2203 | 0.7353 | 0.4099 |

Random Forest achieved the highest test F1, ROC-AUC, and PR-AUC among the Stage-0 models.

The neural network achieved a similar F1 score, while Logistic Regression obtained the highest accuracy but substantially lower recall.

Although the feature set includes TDA-derived descriptors, these experiments do not isolate their incremental contribution. A separate ablation study would be required to determine the predictive value of the topological features.

These results provide the baseline for subsequent deep-learning experiments.

### Stage 1 — Recurrent Neural Networks

**Status: Completed**

Stage 1 investigates whether recurrent neural networks can learn useful temporal dependencies in financial returns and volatility.

The experiments are developed primarily in:

- `10_sequence_models.ipynb`
- `11_sequence_models_rnn.ipynb`

The first notebook explores classical sequence-modeling ideas, including autoregressive and moving-average approaches.

The second notebook investigates three recurrent neural-network architectures:

1. Simple RNN
2. Long Short-Term Memory (LSTM)
3. Gated Recurrent Unit (GRU)

The recurrent models use three input features:

- `log_return`
- `squared_return`
- `vol_20d`

We investigate the effects of lookback-window length, hidden units, dropout regularization, and model architecture.

The experiments also include controlled comparisons between LSTM and GRU models under matched configurations.

#### Selected GRU Configuration

Based on validation performance, the following GRU configuration was selected:

| Hyperparameter | Value |
|---|---|
| Architecture | GRU |
| Hidden units | 16 |
| Lookback window | 60 trading days |
| Dropout | 0.3 |
| Recurrent dropout | 0.0 |
| Classification threshold | 0.25 |
| Random seeds | 42, 123, 2026 |

The classification threshold was selected using validation data before evaluating the model on the held-out test set.

#### Held-Out Test Results

The selected GRU configuration was evaluated across three independently trained models.

| Metric | Mean | Standard Deviation |
|---|---:|---:|
| Accuracy | 0.6453 | 0.0115 |
| Balanced Accuracy | 0.7109 | 0.0024 |
| Precision | 0.3486 | 0.0060 |
| Recall | 0.8228 | 0.0199 |
| F1 | **0.4896** | 0.0030 |
| ROC-AUC | **0.7773** | 0.0086 |
| PR-AUC | **0.4257** | 0.0078 |
| Log Loss | 0.4437 | 0.0027 |

These are averages of independently evaluated models, not metrics obtained from an ensemble.

#### Comparison with Stage 0

| Metric | Stage-0 Random Forest | Stage-1 GRU |
|---|---:|---:|
| Accuracy | 0.5887 | **0.6453** |
| Balanced Accuracy | 0.6765 | **0.7109** |
| Precision | 0.3127 | **0.3486** |
| Recall | **0.8263** | 0.8228 |
| F1 | 0.4538 | **0.4896** |
| ROC-AUC | 0.7557 | **0.7773** |
| PR-AUC | 0.4217 | **0.4257** |
| Log Loss | 0.4538 | **0.4437** |

The selected GRU improves mean test F1 by approximately 7.9% relative to the Stage-0 Random Forest baseline.

It also achieves higher balanced accuracy, ROC-AUC, and PR-AUC, with lower log loss.

However, these improvements are modest, and statistical significance has not been established.

The results suggest that sequential representations contain useful predictive information beyond the fixed-feature baseline, although the experiments do not isolate the contribution of architecture from differences in input representation.

#### Saved Models

The selected GRU models are preserved in `models/gru_final/`.

The saved artifacts include:

- Three trained Keras models corresponding to different random seeds.
- The training-fitted `StandardScaler`.
- Model hyperparameters and inference configuration.
- The selected classification threshold.
- Held-out test evaluation results.

These artifacts support future inference without retraining, provided that the original feature-engineering and preprocessing conventions are reproduced.

### Stage 2 — One-Dimensional Convolutional Neural Networks

**Status: Planned**

Stage 2 will investigate whether one-dimensional convolutional neural networks can learn useful local temporal structure directly from financial time series.

Rather than relying entirely on manually engineered features, the models will operate on sequences of returns or related financial quantities.

Questions of interest include:

- Can a 1D CNN learn useful local patterns in financial time series?
- How does it compare with models based on engineered features?
- What role does temporal locality play in volatility prediction?
- Can learned representations reveal structure not captured by conventional feature engineering?

This stage will provide another perspective on sequence learning and a transition toward structure-aware modeling.

### Stage 3 — Data Augmentation and Geometric Deep Learning

**Status: Planned**

The central motivation for this stage comes from the internship project.

The financial dataset was relatively small and low-to-moderate dimensional, and deep-learning models did not perform as well as the leading tree-based model, XGBoost.

This motivates the question:

> Can symmetry of the prediction function, together with the topology and distribution of the data, be used to construct useful data augmentation?

The goal is not simply to generate additional observations.

An augmentation should be scientifically justified from three complementary perspectives:

1. **Symmetry of the prediction function:** Identify transformations under which the prediction function is invariant or transforms in a known way.

2. **Distribution of the data:** Determine whether transformed observations remain compatible with the statistical distribution of the original financial data.

3. **Topology and geometry of the data:** Determine whether transformations preserve or appropriately modify the geometric and topological structure of the observations.

Topological Data Analysis provides tools for studying the effect of candidate augmentations on data structure, while Geometric Deep Learning provides a framework for thinking about symmetry, invariance, and equivariance.

The objective is to test whether mathematically motivated augmentations can improve predictive performance relative to training on the original dataset alone.

### Stage 4 — Renormalization-Group Flow

**Status: Exploratory**

Stage 4 explores renormalization-group ideas for financial time-series dynamics.

The central question is how the statistical, geometric, and topological structure of the system changes under transformations of scale.

Possible directions include:

- Temporal coarse-graining.
- Multiscale representations.
- Evolution of return distributions under coarse-graining.
- Volatility across scales.
- Evolution of topological quantities across scales.
- Relationships between coarse-grained representations and prediction.

This stage will be developed after the earlier stages have established the relevant empirical and geometric structure.

---

## Mathematical Themes

The project brings together:

- Time-series analysis.
- Volatility modeling.
- Machine learning and deep learning.
- Recurrent and convolutional neural networks.
- Geometric deep learning.
- Persistent homology and topological data analysis.
- Symmetry and invariance.
- Data augmentation.
- Stochastic processes.
- Renormalization-group methods.

A guiding principle is to study statistical, geometric, and topological structure as complementary aspects of the same learning problem.

---

## Topological Data Analysis

The TDA component uses:

- Delay-coordinate embeddings.
- Sliding windows.
- Persistent homology.
- Persistence diagrams.
- Persistence landscapes.
- Landscape norms.

The implementation uses `ripser` and `persim`.

The mathematical and computational background for TDA is developed further in the accompanying repository:

https://github.com/vaishampayan1729/topological-data-analysis

The financial TDA analysis is motivated in part by:

> M. Gidea and Y. Katz, "Topological Data Analysis of Financial Time Series: Landscapes of Crashes," Physica A: Statistical Mechanics and its Applications, vol. 491, pp. 820–834, 2018.

The present project does not attempt to reproduce that study exactly. Instead, it adapts the general idea of extracting topological information from financial time series to a single publicly available asset.

---

## Environment and Reproducibility

The project uses a dedicated Conda environment.

Install the required packages with:

```bash
pip install -r requirements.txt
```

The notebooks document the experimental workflow, including data preparation, feature construction, chronological splits, model training, hyperparameter selection, and evaluation.

Saved model artifacts can be used to reproduce predictions without retraining, subject to compatible software versions and consistent preprocessing.

Raw and processed market datasets are not included in the repository.

---

## Project Status

| Stage | Research Direction | Status |
|---|---|---|
| 0 | Baseline volatility prediction | Completed |
| 1 | Recurrent neural networks | Completed |
| 2 | One-dimensional CNNs | Planned |
| 3 | Data augmentation and geometric deep learning | Planned |
| 4 | Renormalization-group flow | Exploratory |

The next planned research direction is Stage 2: investigating one-dimensional convolutional neural networks for volatility prediction.
