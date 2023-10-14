# Diffusion Model for Time Series and SpatioTemporal Data
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) 
![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green) 
![Stars](https://img.shields.io/github/stars/yyysjz1997/Awesome-TimeSeries-SpatioTemporal-Diffusion-Model)
[![Visits Badge](https://badges.pufler.dev/visits/yyysjz1997/Awesome-TimeSeries-SpatioTemporal-Diffusion-Model)](https://badges.pufler.dev/visits/yyysjz1997/Awesome-TimeSeries-SpatioTemporal-Diffusion-Model)


A curated list of Diffusion Models for Time Series and SpatioTemporal Data with awesome resources (paper, application, review, survey, etc.), which aims to comprehensively and systematically summarize the recent advances to the best of our knowledge.

We will continue to update this list with newest resources. If you found any missed resources (paper/code) or errors, please feel free to open an issue or make a pull request.


## Contents

- [Diffusion Model for Time Series and SpatioTemporal Data](#diffusion-model-for-time-series-and-spatiotemporal-data)
  - [Contents](#contents)
  - [Diffusion Model for Time Series Data](#diffusion-model-for-time-series-data)
    - [Generation](#generation)
    - [Anomaly Detection](#anomaly-detection)
    - [Forecasting and Prediction](#forecasting-and-prediction)
    - [Imputation](#imputation)
    - [Classification and Regression](#classification-and-regression)
    - [Causal Inference](#causal-inference)
  - [Diffusion Model for SpatioTemporal Data](#diffusion-model-for-spatiotemporal-data)
  - [Diffusion Model for Tabular Data](#diffusion-model-for-tabular-data)
  - [Applications](#applications)
    - [Energy and Electricity](#energy-and-electricity)
    - [Finance](#finance)
    - [Healthcare](#healthcare)
    - [Weather](#weather)
    - [AIOps](#aiops)
    - [Sequential Recommendation](#sequential-recommendation)
    - [Human](#human)
    - [Environment](#environment)
    - [Math and Physics](#math-and-physics)
  - [Related Diffusion Model Surveys](#related-diffusion-model-surveys)



## Diffusion Model for Time Series Data

### Generation

* Conditioning Score-Based Generative Models by Neuro-Symbolic Constraints, in *arXiv* 2023. [[Paper](https://arxiv.org/abs/2308.16534)]

* TransFusion: Generating Long, High Fidelity Time Series using Diffusion Models with Transformers, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2307.12667)]

* On the Constrained Time-Series Generation Problem, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2307.01717)]

* DiffECG: A Generalized Probabilistic Diffusion Model for ECG Signals Synthesis, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2306.01875)]

* Customized Load Profiles Synthesis for Electricity Customers Based on Conditional Diffusion Models, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2304.12076)]

* Synthetic Health-related Longitudinal Data with Mixed-type Variables Generated using Diffusion Models, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2303.12281)]

* Diffusing Gaussian Mixtures for Generating Categorical Data, in *AAAI* 2023. [[paper](https://arxiv.org/abs/2303.04635)] [[official-code](https://github.com/networkslab/gmcd)]

* EHRDiff: Exploring Realistic EHR Synthesis with Diffusion Models, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2303.05656)] [[official-code](https://github.com/sczzz3/ehrdiff)]

* Synthesizing Mixed-type Electronic Health Records using Diffusion Models, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2302.14679)]

* MedDiff: Generating Electronic Health Records using Accelerated Denoising Diffusion Model, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2302.04355)]

* WaveGrad: Estimating Gradients for Waveform Generation, in *ICLR* 2021. [[paper](https://arxiv.org/abs/2009.00713)]

* Diffusion Generative Models in Infinite Dimensions, in *AISTATS* 2023. [[paper](https://arxiv.org/abs/2212.00886)] [[official-code](https://github.com/gavinkerrigan/functional_diffusion)]


### Anomaly Detection

* Imputation-based Time-Series Anomaly Detection with Conditional Weight-Incremental Diffusion Models, in *KDD* 2023. [[paper](https://dl.acm.org/doi/abs/10.1145/3580305.3599391)] [[official-code](https://github.com/ChunjingXiao/DiffAD)]

* ImDiffusion: Imputed Diffusion Models for Multivariate Time Series Anomaly Detection, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2307.00754)] [[official-code](https://github.com/17000cyh/IMDiffusion)]


### Forecasting and Prediction

* DiffECG: A Generalized Probabilistic Diffusion Model for ECG Signals Synthesis, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2306.01875)]

* Diffusion Variational Autoencoder for Tackling Stochasticity in Multi-Step Regression Stock Price Prediction, in *CIKM* 2023. [[paper](https://arxiv.org/abs/2309.00073)] [[official-code](https://github.com/koa-fin/dva)]

* Predict, Refine, Synthesize: Self-Guiding Diffusion Models for Probabilistic Time Series Forecasting, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2307.11494)]

* Data Augmentation for Seizure Prediction with Generative Diffusion Model, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2306.08256)]

* Non-autoregressive Conditional Diffusion Models for Time Series Prediction, in *ICML* 2023. [[paper](https://dl.acm.org/doi/10.5555/3618408.3619692)]

* DiffLoad: Uncertainty Quantification in Load Forecasting with Diffusion Model, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2306.01001)]

* Generative Time Series Forecasting with Diffusion, Denoise, and Disentanglement, in *NeurIPS* 2022. [[paper](https://arxiv.org/abs/2301.03028)] [[official-code](https://github.com/PaddlePaddle/PaddleSpatial/tree/main/research/D3VAE)]

* Denoising Diffusion Probabilistic Models for Probabilistic Energy Forecasting, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2212.02977)] [[official-code](https://github.com/EstebanHernandezCapel/DDPM-Power-systems-forecasting)]

* Modeling Temporal Data as Continuous Functions with Process Diffusion, in *ICML* 2023. [[paper](https://arxiv.org/abs/2211.02590)]

* ScoreGrad: Multivariate Probabilistic Time Series Forecasting with Continuous Energy-based Generative Models, in *arXiv* 2021. [[paper](https://arxiv.org/abs/2106.10121)] [[official-code](https://github.com/yantijin/ScoreGradPred)]

* TDSTF: Transformer-based Diffusion probabilistic model for Sparse Time series Forecasting, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2301.06625)] [[official-code](https://github.com/pingchang818/tdstf)]

* Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting, in *ICML* 2021. [[paper](https://arxiv.org/abs/2101.12072)] [[official-code](https://github.com/zalandoresearch/pytorch-ts)]

* Graph Convolution Recurrent Denoising Diffusion Model for Multivariate Probabilistic Temporal Forecasting, *Working Paper* 2023. [[link](https://aip.riken.jp/events/event_154762/)]


### Imputation

* Modeling Temporal Data as Continuous Functions with Process Diffusion, in *ICML* 2023. [[paper](https://arxiv.org/abs/2211.02590)]

* CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation, in *NeurIPS* 2021. [[paper](https://arxiv.org/abs/2107.03502)] [[official-code](https://github.com/ermongroup/CSDI)]

* An Observed Value Consistent Diffusion Model for Imputing Missing Values in Multivariate Time Series, in *KDD* 2023. [[paper](https://dl.acm.org/doi/10.1145/3580305.3599257)]

* Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models, in *Transactions on Machine Learning Research (TMLR)* 2023. [[paper](https://arxiv.org/abs/2208.09399)] [[official-code](https://github.com/ai4healthuol/sssd)]

* Sasdim: Self-adaptive Noise Scaling Diffusion Model for Spatial Time Series Imputation, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2309.01988)]


### Classification and Regression

* CARD: Classification and Regression Diffusion Models, in *NeurIPS* 2022. [[paper](https://arxiv.org/abs/2206.07275)] [[official-code](https://github.com/xzwhan/card)]


### Causal Inference

* Diffusion Model in Causal Inference with Unmeasured Confounders, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2308.03669)] [[official-code](https://github.com/tatsu432/BDCM)]



## Diffusion Model for SpatioTemporal Data

* Spatio-temporal Diffusion Point Processes

* DYffusion: A Dynamics-informed Diffusion Model for Spatiotemporal Forecasting

* DiffSTG: Probabilistic Spatio-Temporal Graph with Denoising Diffusion Models

* PriSTI: A Conditional Diffusion Framework for Spatiotemporal Imputation

* Extraction and Recovery of Spatio-Temporal Structure in Latent Dynamics Alignment with Diffusion Model

* Dynamic Causal Explanation Based Diffusion-Variational Graph Neural Network for Spatio-temporal Forecasting

* Imputation as Inpainting: Diffusion Models for Spatiotemporal Data Imputation

* Sasdim: Self-adaptive Noise Scaling Diffusion Model for Spatial Time Series Imputation


## Diffusion Model for Tabular Data

* [Arxiv 2023] Conditioning Score-Based Generative Models by Neuro-Symbolic Constraints. [Paper](https://arxiv.org/abs/2308.16534)

* On Diffusion Modeling for Anomaly Detection, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2305.18593)] [[official-code](https://github.com/vicliv/dte)]

* Generating Tabular Datasets under Differential Privacy

* TabADM: Unsupervised Tabular Anomaly Detection with Diffusion Models

* CoDi: Co-evolving Contrastive Diffusion Models for Mixed-type Tabular Synthesis

* TabDDPM: Modelling Tabular Data with Diffusion Models

* Diffusion Models for Missing Value Imputation in Tabular Data

* MissDiff: Training Diffusion Models on Tabular Data with Missing Values

* FinDiff: Diffusion Models for Financial Tabular Data Generation


## Applications

### Energy and Electricity

* DiffCharge: Generating EV Charging Scenarios via a Denoising Diffusion Model

* Customized Load Profiles Synthesis for Electricity Customers Based on Conditional Diffusion Models

* DiffLoad: Uncertainty Quantification in Load Forecasting with Diffusion Model

* Denoising Diffusion Probabilistic Models for Probabilistic Energy Forecasting


### Finance

* Option Pricing Models Driven by the Space-Time Fractional Diffusion: Series Representation and Applications

* FinDiff: Diffusion Models for Financial Tabular Data Generation

* Diffusion Variational Autoencoder for Tackling Stochasticity in Multi-Step Regression Stock Price Prediction


### Healthcare

* DiffECG: A Generalized Probabilistic Diffusion Model for ECG Signals Synthesis

* Synthetic Health-related Longitudinal Data with Mixed-type Variables Generated using Diffusion Models

* EHRDiff: Exploring Realistic EHR Synthesis with Diffusion Models

* Synthesizing Mixed-type Electronic Health Records using Diffusion Models

* MedDiff: Generating Electronic Health Records using Accelerated Denoising Diffusion Model

* Data Augmentation for Seizure Prediction with Generative Diffusion Model

* Region-Disentangled Diffusion Model for High-Fidelity PPG-to-ECG Translation

* Diff-E: Diffusion-based Learning for Decoding Imagined Speech EEG

* EEG Synthetic Data Generation Using Probabilistic Diffusion Models

* DeScoD-ECG: Deep Score-Based Diffusion Model for ECG Baseline Wander and Noise Removal

* Diffusion-based Conditional ECG Generation with Structured State Space Models

* Domain-Specific Denoising Diffusion Probabilistic Models for Brain Dynamics


### Weather

* Precipitation Nowcasting with Generative Diffusion Models

* SEEDS: Emulation of Weather Forecast Ensembles with Diffusion Models

* SwinRDM: Integrate SwinRNN with Diffusion Model towards High-Resolution and High-Quality Weather Forecasting

* DiffESM: Conditional Emulation of Earth System Models with Diffusion Models

* Diffusion Models for High-Resolution Solar Forecasts


### AIOps

* Maat: Performance Metric Anomaly Anticipation for Cloud Services with Conditional Diffusion

* NetDiffus: Network Traffic Generation by Diffusion Models through Time-Series Imaging


### Sequential Recommendation

* Discrete Conditional Diffusion for Reranking in Recommendation

* RecFusion: A Binomial Diffusion Process for 1D Data for Recommendation

* Conditional Denoising Diffusion for Sequential Recommendation

* Diffusion Recommender Model

* Sequential Recommendation with Diffusion Models

* DiffuRec: A Diffusion Model for Sequential Recommendation

* Recommendation via Collaborative Diffusion Generative Model


### Human

* Unsupervised Statistical Feature-Guided Diffusion Model for Sensor-based Human Activity Recognition


### Environment

* Deep Diffusion Models for Seismic Processing


### Math and Physics

* DiTTO: Diffusion-inspired Temporal Transformer Operator

* Fast Sampling of Diffusion Models via Operator Learning

* Score-based Diffusion Models in Function Space

* Infinite-dimensional Diffusion Models for Function Spaces

* A Physics-informed Diffusion Model for High-fidelity Flow Field Reconstruction

* Generative Diffusion Learning for Parametric Partial Differential Equations



## Related Diffusion Model Surveys

* Diffusion Models for Time Series Applications: A Survey, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2305.00624)]

* A Comprehensive Survey on Generative Diffusion Models for Structured Data, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2306.04139)] [[link](https://deepai.org/publication/a-survey-on-generative-diffusion-models-for-structured-data)]

* Diffusion Models: A Comprehensive Survey of Methods and Applications, in *ACM Computing Surveys* 2023. [[paper](https://arxiv.org/abs/2209.00796)] [[link](https://github.com/chq1155/A-Survey-on-Generative-Diffusion-Model)]

* A Survey on Generative Diffusion Models, in *arXiv* 2022. [[paper](https://arxiv.org/abs/2209.02646)] [[link](https://github.com/YangLing0818/Diffusion-Models-Papers-Survey-Taxonomy)]

* Awesome-Diffusion-Models. [[link](https://diff-usion.github.io/Awesome-Diffusion-Models/)]
