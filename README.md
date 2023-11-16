# Diffusion Model for Time Series, SpatioTemporal and Tabular Data

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) 
![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green) 
![Stars](https://img.shields.io/github/stars/yyysjz1997/Awesome-TimeSeries-SpatioTemporal-Diffusion-Model)
[![Visits Badge](https://badges.pufler.dev/visits/yyysjz1997/Awesome-TimeSeries-SpatioTemporal-Diffusion-Model)](https://badges.pufler.dev/visits/yyysjz1997/Awesome-TimeSeries-SpatioTemporal-Diffusion-Model)


A curated list of **Diffusion Models** for **Time Series**, **SpatioTemporal Data** and **Tabular Data** with awesome resources (paper, code, application, review, survey, etc.), which aims to comprehensively and systematically summarize the recent advances to the best of our knowledge.

We will continue to update this list with newest resources. If you found any missed resources (paper/code) or errors, please feel free to open an issue or make a pull request.



## Contents

- [Diffusion Model for Time Series and SpatioTemporal Data](#diffusion-model-for-time-series-and-spatiotemporal-data)
  - [Contents](#contents)
  - [Diffusion Model for Time Series Data](#diffusion-model-for-time-series-data)
    - [Forecasting and Prediction](#forecasting-and-prediction)
    - [Generation](#generation)
    - [Imputation](#imputation)
    - [Anomaly Detection](#anomaly-detection)
    - [Classification and Regression](#classification-and-regression)
    - [Causal Inference](#causal-inference)
    - [Event Prediction](#event-prediction)
  - [Diffusion Model for SpatioTemporal Data](#diffusion-model-for-spatiotemporal-data)
  - [Diffusion Model for Tabular Data](#diffusion-model-for-tabular-data)
  - [Applications](#applications)
    - [Healthcare](#healthcare)
    - [Sequential Recommendation](#sequential-recommendation)
    - [Weather](#weather)
    - [Video](#video)
    - [Energy and Electricity](#energy-and-electricity)
    - [Math and Physics](#math-and-physics)
    - [Finance](#finance)
    - [AIOps](#aiops)
    - [Environment](#environment)
    - [Traffic](#traffic)
    - [Manufacturing](#manufacturing)
    - [Audio](#audio)
  - [Related Diffusion Model Resources and Surveys](#related-diffusion-model-resources-and-surveys)



## Diffusion Model for Time Series Data

### Forecasting and Prediction

* Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting, in *ICML* 2021. [[paper](https://arxiv.org/abs/2101.12072)] [[official-code](https://github.com/zalandoresearch/pytorch-ts)]

* ScoreGrad: Multivariate Probabilistic Time Series Forecasting with Continuous Energy-based Generative Models, in *arXiv* 2021. [[paper](https://arxiv.org/abs/2106.10121)] [[official-code](https://github.com/yantijin/ScoreGradPred)]

* Generative Time Series Forecasting with Diffusion, Denoise, and Disentanglement, in *NeurIPS* 2022. [[paper](https://arxiv.org/abs/2301.03028)] [[official-code](https://github.com/PaddlePaddle/PaddleSpatial/tree/main/research/D3VAE)]

* Diffusion Variational Autoencoder for Tackling Stochasticity in Multi-Step Regression Stock Price Prediction, in *CIKM* 2023. [[paper](https://arxiv.org/abs/2309.00073)] [[official-code](https://github.com/koa-fin/dva)]

* Modeling Temporal Data as Continuous Functions with Process Diffusion, in *ICML* 2023. [[paper](https://arxiv.org/abs/2211.02590)]

* Non-autoregressive Conditional Diffusion Models for Time Series Prediction, in *ICML* 2023. [[paper](https://dl.acm.org/doi/10.5555/3618408.3619692)]

* Graph Convolution Recurrent Denoising Diffusion Model for Multivariate Probabilistic Temporal Forecasting, in *International Conference on Advanced Data Mining and Applications (ADMA)* 2023. [[paper](https://dl.acm.org/doi/10.5555/3618408.3619692)]

* Generative Time Series Forecasting with Diffusion, Denoise, and Disentanglement, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2301.03028)] [[official-code](https://github.com/paddlepaddle/paddlespatial)]

* DiffECG: A Generalized Probabilistic Diffusion Model for ECG Signals Synthesis, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2306.01875)]

* Predict, Refine, Synthesize: Self-Guiding Diffusion Models for Probabilistic Time Series Forecasting, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2307.11494)]

* Data Augmentation for Seizure Prediction with Generative Diffusion Model, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2306.08256)]

* DiffLoad: Uncertainty Quantification in Load Forecasting with Diffusion Model, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2306.01001)]

* Denoising Diffusion Probabilistic Models for Probabilistic Energy Forecasting, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2212.02977)] [[official-code](https://github.com/EstebanHernandezCapel/DDPM-Power-systems-forecasting)]

* TDSTF: Transformer-based Diffusion probabilistic model for Sparse Time series Forecasting, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2301.06625)] [[official-code](https://github.com/pingchang818/tdstf)]

* Graph Convolution Recurrent Denoising Diffusion Model for Multivariate Probabilistic Temporal Forecasting, *Working Paper* 2023. [[link](https://aip.riken.jp/events/event_154762/)]

* Latent Diffusion Models for Generative Precipitation Nowcasting with Accurate Uncertainty Quantification, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2304.12891)] [[official-code](https://github.com/MeteoSwiss/ldcast)]

* WiREDiff: a Wind Resolution-Enhancing Diffusion Model, in *Working Paper* 2023. [[link](https://rupakv.com/wirediff.html#:~:text=WiREDiff%20is%20a%20novel%20application,interval%20of%20predicted%20wind%20velocities.)] [[paper](https://rupakv.com/pdfs/wirediff_report.pdf)] [[official-code](https://github.com/RupaKurinchiVendhan/WiREDiff)]


### Generation

* WaveGrad: Estimating Gradients for Waveform Generation, in *ICLR* 2021. [[paper](https://arxiv.org/abs/2009.00713)]

* DiffWave: A Versatile Diffusion Model for Audio Synthesis, in *ICLR* 2021. [[paper](https://arxiv.org/abs/2009.09761)] [[official-code](https://diffwave-demo.github.io/)]

* Diffusing Gaussian Mixtures for Generating Categorical Data, in *AAAI* 2023. [[paper](https://arxiv.org/abs/2303.04635)] [[official-code](https://github.com/networkslab/gmcd)]

* Diffusion Generative Models in Infinite Dimensions, in *AISTATS* 2023. [[paper](https://arxiv.org/abs/2212.00886)] [[official-code](https://github.com/gavinkerrigan/functional_diffusion)]

* Multi-scale Conditional Diffusion Model for Deposited Droplet Volume Measurement in Inkjet Printing Manufacturing, in *Journal of Manufacturing Systems* 2023. [[paper](https://www.sciencedirect.com/science/article/pii/S0278612523002042)]
 
* Diffusion Augmentation for Sequential Recommendation, in *CIKM* 2023. [[paper](https://dl.acm.org/doi/abs/10.1145/3583780.3615134)] [[official-code](https://github.com/liuqidong07/DiffuASR)]

* Short-Term Wind Power Scenario Generation Based on Conditional Latent Diffusion Models, in *IEEE Transactions on Sustainable Energy* 2023. [[paper](https://ieeexplore.ieee.org/abstract/document/10296015)]

* Synthetic Sleep EEG Signal Generation using Latent Diffusion Models, in *DGM4H NeurIPS* 2023. [[paper](https://openreview.net/forum?id=mDwURmlapW)] [[official-code](https://github.com/bruAristimunha/Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models)]

* Conditioning Score-Based Generative Models by Neuro-Symbolic Constraints, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2308.16534)]

* TransFusion: Generating Long, High Fidelity Time Series using Diffusion Models with Transformers, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2307.12667)]

* On the Constrained Time-Series Generation Problem, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2307.01717)]

* DiffECG: A Generalized Probabilistic Diffusion Model for ECG Signals Synthesis, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2306.01875)]

* Customized Load Profiles Synthesis for Electricity Customers Based on Conditional Diffusion Models, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2304.12076)]

* Synthetic Health-related Longitudinal Data with Mixed-type Variables Generated using Diffusion Models, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2303.12281)]

* EHRDiff: Exploring Realistic EHR Synthesis with Diffusion Models, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2303.05656)] [[official-code](https://github.com/sczzz3/ehrdiff)]

* Synthesizing Mixed-type Electronic Health Records using Diffusion Models, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2302.14679)]

* MedDiff: Generating Electronic Health Records using Accelerated Denoising Diffusion Model, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2302.04355)]

* Fast and Reliable Generation of EHR Time Series via Diffusion Models, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2310.15290)]

* Fast Diffusion GAN Model for Symbolic Music Generation Controlled by Emotions, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2310.14040)]


### Imputation

* CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation, in *NeurIPS* 2021. [[paper](https://arxiv.org/abs/2107.03502)] [[official-code](https://github.com/ermongroup/CSDI)]
  
* Modeling Temporal Data as Continuous Functions with Process Diffusion, in *ICML* 2023. [[paper](https://arxiv.org/abs/2211.02590)]

* An Observed Value Consistent Diffusion Model for Imputing Missing Values in Multivariate Time Series, in *KDD* 2023. [[paper](https://dl.acm.org/doi/10.1145/3580305.3599257)]

* Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models, in *Transactions on Machine Learning Research (TMLR)* 2023. [[paper](https://arxiv.org/abs/2208.09399)] [[official-code](https://github.com/ai4healthuol/sssd)]

* Density-Aware Temporal Attentive Step-wise Diffusion Model For Medical Time Series Imputation, in *CIKM* 2023. [[paper](https://dl.acm.org/doi/abs/10.1145/3583780.3614840)]

* MEDiC: Mitigating EEG Data Scarcity Via Class-Conditioned Diffusion Model, in *DGM4H NeurIPS* 2023. [[paper](https://openreview.net/forum?id=0aeDKGhlTo)]

* Missing Data Filling in Soft Sensing Using Denoising Diffusion Probability Model, in *Measurement Science and Technology* 2023. [[paper](https://iopscience.iop.org/article/10.1088/1361-6501/ad095a)]

* Sasdim: Self-adaptive Noise Scaling Diffusion Model for Spatial Time Series Imputation, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2309.01988)]

* Improving Diffusion Models for ECG Imputation with an Augmented Template Prior, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2310.15742)]


### Anomaly Detection

* Imputation-based Time-Series Anomaly Detection with Conditional Weight-Incremental Diffusion Models, in *KDD* 2023. [[paper](https://dl.acm.org/doi/abs/10.1145/3580305.3599391)] [[official-code](https://github.com/ChunjingXiao/DiffAD)]

* Time Series Anomaly Detection using Diffusion-based Models, in *ICDM Workshop* 2023. [[paper](https://arxiv.org/abs/2311.01452)] [[official-code](https://github.com/fbrad/diffusionae)]

* Drift doesn’t Matter: Dynamic Decomposition with Diffusion Reconstruction for Unstable Multivariate Time Series Anomaly Detection, in *NeurIPS* 2023. [[paper](https://openreview.net/pdf?id=aW5bSuduF1)] [[official-code](https://github.com/ForestsKing/D3R)]

* ImDiffusion: Imputed Diffusion Models for Multivariate Time Series Anomaly Detection, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2307.00754)] [[official-code](https://github.com/17000cyh/IMDiffusion)]

* DDMT: Denoising Diffusion Mask Transformer Models for Multivariate Time Series Anomaly Detection, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2310.08800)] [[official-code](https://github.com/yangchaocheng/DDTM)]

* Generating HSR Bogie Vibration Signals via Pulse Voltage-Guided Conditional Diffusion Model, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2311.00496)] [[official-code](https://github.com/xuanliu2000/VGCDM)]


### Classification and Regression

* CARD: Classification and Regression Diffusion Models, in *NeurIPS* 2022. [[paper](https://arxiv.org/abs/2206.07275)] [[official-code](https://github.com/xzwhan/card)]

* Applying Generative Models and Transfer Learning to Physiological Data Classification, in *Artificial Intelligence Research and Development* 2023. [[paper](https://ebooks.iospress.nl/doi/10.3233/FAIA230656)]


### Causal Inference

* Diffusion Model in Causal Inference with Unmeasured Confounders, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2308.03669)] [[official-code](https://github.com/tatsu432/BDCM)]


### Event Prediction

* Non-Autoregressive Diffusion-based Temporal Point Processes for Continuous-Time Long-Term Event Prediction, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2311.01033)]

* Add and Thin: Diffusion for Temporal Point Processes, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2311.01139)] [[official-code](https://www.cs.cit.tum.de/daml/add-thin/)]



## Diffusion Model for SpatioTemporal Data

* Spatio-temporal Diffusion Point Processes, in *KDD* 2023. [[paper](https://arxiv.org/abs/2305.12403)] [[official-code](https://github.com/tsinghua-fib-lab/Spatio-temporal-Diffusion-Point-Processes)]

* DYffusion: A Dynamics-informed Diffusion Model for Spatiotemporal Forecasting, in *NeurIPS* 2023. [[paper](https://arxiv.org/abs/2306.01984)] [[official-code](https://github.com/Rose-STL-Lab/dyffusion)]

* Extraction and Recovery of Spatio-Temporal Structure in Latent Dynamics Alignment with Diffusion Model, in *NeurIPS* 2023. [[paper](https://arxiv.org/abs/2306.06138)] [[official-code](https://github.com/alexwangntl/erdiff)]

* DiffSTG: Probabilistic Spatio-Temporal Graph with Denoising Diffusion Models, in *SIGSPATIAL* 2023. [[paper](https://arxiv.org/abs/2301.13629)]

* PriSTI: A Conditional Diffusion Framework for Spatiotemporal Imputation, in *ICDE* 2023. [[paper](https://arxiv.org/abs/2302.09746)] [[official-code](https://github.com/lmzzml/pristi)]

* DiffUFlow: Robust Fine-grained Urban Flow Inference with Denoising Diffusion Model, in *CIKM* 2023. [[paper](https://dl.acm.org/doi/abs/10.1145/3583780.3614842)]

* Adaptive Graph Neural Diffusion for Traffic Demand Forecasting, in *CIKM* 2023. [[paper](https://dl.acm.org/doi/10.1145/3583780.3615153)]

* DiffTraj: Generating GPS Trajectory with Diffusion Probabilistic Model, in *NeurIPS* 2023. [[paper](https://arxiv.org/abs/2304.11582)] [[official-code](https://github.com/Yasoz/DiffTraj)]

* Dynamic Causal Explanation Based Diffusion-Variational Graph Neural Network for Spatio-temporal Forecasting, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2305.09703)] [[official-code](https://github.com/gorgen2020/dvgnn)]

* Sasdim: Self-adaptive Noise Scaling Diffusion Model for Spatial Time Series Imputation, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2309.01988)]

* Imputation as Inpainting: Diffusion Models for Spatiotemporal Data Imputation, in *OpenReview* 2023. [[paper](https://openreview.net/forum?id=QUANtQnx30l)]

* Towards Unifying Diffusion Models for Probabilistic Spatio-Temporal Graph Learning, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2310.17360)]

* Predicting the Position Uncertainty at the Time of Closest Approach with Diffusion Models, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2311.05417)]



## Diffusion Model for Tabular Data

* Diffusion Models for Missing Value Imputation in Tabular Data, in *NeurIPS TRL Workshop* 2022. [[paper](https://arxiv.org/abs/2210.17128)] [[official-code](https://github.com/pfnet-research/TabCSDI)]

* MissDiff: Training Diffusion Models on Tabular Data with Missing Values, in *ICML Workshop* 2023. [[paper](https://arxiv.org/abs/2307.00467)]

* CoDi: Co-evolving Contrastive Diffusion Models for Mixed-type Tabular Synthesis, in *ICML* 2023. [[paper](https://arxiv.org/abs/2304.12654)] [[official-code](https://github.com/chaejeonglee/codi)]

* TabDDPM: Modelling Tabular Data with Diffusion Models, in *ICML* 2023. [[paper](https://arxiv.org/abs/2209.15421)] [[official-code](https://github.com/yandex-research/tab-ddpm)]
 
* Conditioning Score-Based Generative Models by Neuro-Symbolic Constraints, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2308.16534)]

* On Diffusion Modeling for Anomaly Detection, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2305.18593)] [[official-code](https://github.com/vicliv/dte)]

* Generating Tabular Datasets under Differential Privacy, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2308.14784)]

* TabADM: Unsupervised Tabular Anomaly Detection with Diffusion Models, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2307.12336)]

* FinDiff: Diffusion Models for Financial Tabular Data Generation, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2309.01472)]

* AutoDiff: combining Auto-encoder and Diffusion model for tabular data synthesizing, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2310.15479)]

  


## Applications

### Healthcare

* Diff-E: Diffusion-based Learning for Decoding Imagined Speech EEG, in *Interspeech* 2023. [[paper](https://arxiv.org/abs/2307.14389)] [[official-code](https://github.com/yorgoon/diffe)]

* EEG Synthetic Data Generation Using Probabilistic Diffusion Models, in *Synapsium* 2023. [[paper](https://arxiv.org/abs/2303.06068)] [[official-code](https://github.com/devjake/eeg-diffusion-pytorch)]

* DeScoD-ECG: Deep Score-Based Diffusion Model for ECG Baseline Wander and Noise Removal, in *IEEE Journal of Biomedical and Health Informatics* 2023. [[paper](https://arxiv.org/abs/2208.00542)] [[official-code](https://github.com/huayuliarizona/score-based-ecg-denoising)]

* Diffusion-based Conditional ECG Generation with Structured State Space Models, in *Computers in Biology and Medicine* 2023. [[paper](https://arxiv.org/abs/2301.08227)] [[official-code](https://github.com/ai4healthuol/sssd-ecg)]

* Density-Aware Temporal Attentive Step-wise Diffusion Model For Medical Time Series Imputation, in *CIKM* 2023. [[paper](https://dl.acm.org/doi/abs/10.1145/3583780.3614840)]

* Applying Generative Models and Transfer Learning to Physiological Data Classification, in *Artificial Intelligence Research and Development* 2023. [[paper](https://ebooks.iospress.nl/doi/10.3233/FAIA230656)]

* MEDiC: Mitigating EEG Data Scarcity Via Class-Conditioned Diffusion Model, in *DGM4H NeurIPS* 2023. [[paper](https://openreview.net/forum?id=0aeDKGhlTo)]

* Synthetic Sleep EEG Signal Generation using Latent Diffusion Models, in *DGM4H NeurIPS* 2023. [[paper](https://openreview.net/forum?id=mDwURmlapW)] [[official-code](https://github.com/bruAristimunha/Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models)]

* DiffECG: A Generalized Probabilistic Diffusion Model for ECG Signals Synthesis, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2306.01875)]

* Synthetic Health-related Longitudinal Data with Mixed-type Variables Generated using Diffusion Models, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2303.12281)]

* EHRDiff: Exploring Realistic EHR Synthesis with Diffusion Models, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2303.05656)] [[official-code](https://github.com/sczzz3/ehrdiff)]

* Synthesizing Mixed-type Electronic Health Records using Diffusion Models, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2302.14679)]

* MedDiff: Generating Electronic Health Records using Accelerated Denoising Diffusion Model, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2302.04355)]

* Data Augmentation for Seizure Prediction with Generative Diffusion Model, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2306.08256)]

* Region-Disentangled Diffusion Model for High-Fidelity PPG-to-ECG Translation, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2308.13568)]

* Domain-Specific Denoising Diffusion Probabilistic Models for Brain Dynamics, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2305.04200)] [[official-code](https://github.com/duanyiqun/ds-ddpm)]

* Fast and Reliable Generation of EHR Time Series via Diffusion Models, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2310.15290)]

* Improving Diffusion Models for ECG Imputation with an Augmented Template Prior, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2310.15742)]


### Sequential Recommendation

* Recommendation via Collaborative Diffusion Generative Model, in *Knowledge Science, Engineering and Management: 15th International Conference (KSEM)* 2022. [[paper](https://dl.acm.org/doi/10.1007/978-3-031-10989-8_47)]

* Diffusion Recommender Model, in *SIGIR* 2023. [[paper](https://arxiv.org/abs/2304.04971)] [[official-code](https://github.com/yiyanxu/diffrec)]

* Diffusion Augmentation for Sequential Recommendation, in *CIKM* 2023. [[paper](https://dl.acm.org/doi/abs/10.1145/3583780.3615134)] [[official-code](https://github.com/liuqidong07/DiffuASR)]

* Discrete Conditional Diffusion for Reranking in Recommendation, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2308.06982)]

* RecFusion: A Binomial Diffusion Process for 1D Data for Recommendation, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2306.08947)] [[official-code](https://github.com/gabriben/recfusion)]

* Conditional Denoising Diffusion for Sequential Recommendation, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2304.11433)]

* Sequential Recommendation with Diffusion Models, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2304.04541)]

* DiffuRec: A Diffusion Model for Sequential Recommendation, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2304.00686)]


### Weather

* SwinRDM: Integrate SwinRNN with Diffusion Model towards High-Resolution and High-Quality Weather Forecasting, in *AAAI* 2023. [[paper](https://arxiv.org/abs/2306.03110)]

* DiffESM: Conditional Emulation of Earth System Models with Diffusion Models, in *ICLR* 2023. [[paper](https://arxiv.org/abs/2304.11699)] [[official-code](https://github.com/duanyiqun/ds-ddpm)]

* Precipitation Nowcasting with Generative Diffusion Models, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2308.06733)] [[official-code](https://github.com/fmerizzi/precipitation-nowcasting-with-generative-diffusion-models)]

* SEEDS: Emulation of Weather Forecast Ensembles with Diffusion Models, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2306.14066)]

* Diffusion Models for High-Resolution Solar Forecasts, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2302.00170)]

* Latent Diffusion Models for Generative Precipitation Nowcasting with Accurate Uncertainty Quantification, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2304.12891)] [[official-code](https://github.com/MeteoSwiss/ldcast)]

* WiREDiff: a Wind Resolution-Enhancing Diffusion Model, in *Working Paper* 2023. [[link](https://rupakv.com/wirediff.html#:~:text=WiREDiff%20is%20a%20novel%20application,interval%20of%20predicted%20wind%20velocities.)] [[paper](https://rupakv.com/pdfs/wirediff_report.pdf)] [[official-code](https://github.com/RupaKurinchiVendhan/WiREDiff)]


### Video

*


### Energy and Electricity

* Short-Term Wind Power Scenario Generation Based on Conditional Latent Diffusion Models, in *IEEE Transactions on Sustainable Energy* 2023. [[paper](https://ieeexplore.ieee.org/abstract/document/10296015)]

* DiffCharge: Generating EV Charging Scenarios via a Denoising Diffusion Model, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2308.09857)] [[official-code](https://github.com/LSY-Cython/DiffCharge)]

* Customized Load Profiles Synthesis for Electricity Customers Based on Conditional Diffusion Models, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2304.12076)]

* DiffLoad: Uncertainty Quantification in Load Forecasting with Diffusion Model, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2306.01001)]

* Denoising Diffusion Probabilistic Models for Probabilistic Energy Forecasting, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2212.02977)] [[official-code](https://github.com/EstebanHernandezCapel/DDPM-Power-systems-forecasting)]


### Math and Physics

* A Physics-informed Diffusion Model for High-fidelity Flow Field Reconstruction, in *Journal of Computational Physics* 2023. [[paper](https://arxiv.org/abs/2211.14680)] [[official-code](https://github.com/BaratiLab/Diffusion-based-Fluid-Super-resolution)]

* DiTTO: Diffusion-inspired Temporal Transformer Operator, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2307.09072)]

* Infinite-dimensional Diffusion Models for Function Spaces, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2302.10130)]

* Generative Diffusion Learning for Parametric Partial Differential Equations, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2305.14703)]


### Finance

* Diffusion Variational Autoencoder for Tackling Stochasticity in Multi-Step Regression Stock Price Prediction, in *CIKM* 2023. [[paper](https://arxiv.org/abs/2309.00073)] [[official-code](https://github.com/koa-fin/dva)]

* FinDiff: Diffusion Models for Financial Tabular Data Generation, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2309.01472)]


### AIOps

* Maat: Performance Metric Anomaly Anticipation for Cloud Services with Conditional Diffusion, in *IEEE/ACM International Conference on Automated Software Engineering* 2023. [[paper](https://arxiv.org/abs/2308.07676)] [[official-code](https://github.com/bebillionaireusd/maat)]

* NetDiffus: Network Traffic Generation by Diffusion Models through Time-Series Imaging, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2310.04429)]


### Environment

* Deep Diffusion Models for Seismic Processing, in *Computers & Geosciences* 2023. [[paper](https://arxiv.org/abs/2207.10451)]


### Traffic

* DiffUFlow: Robust Fine-grained Urban Flow Inference with Denoising Diffusion Model, in *CIKM* 2023. [[paper](https://dl.acm.org/doi/abs/10.1145/3583780.3614842)]

* Adaptive Graph Neural Diffusion for Traffic Demand Forecasting, in *CIKM* 2023. [[paper](https://dl.acm.org/doi/10.1145/3583780.3615153)]

* Generating HSR Bogie Vibration Signals via Pulse Voltage-Guided Conditional Diffusion Model, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2311.00496)] [[official-code](https://github.com/xuanliu2000/VGCDM)]


### Manufacturing

* Multi-scale Conditional Diffusion Model for Deposited Droplet Volume Measurement in Inkjet Printing Manufacturing, in *Journal of Manufacturing Systems* 2023. [[paper](https://www.sciencedirect.com/science/article/pii/S0278612523002042)]

* Missing Data Filling in Soft Sensing Using Denoising Diffusion Probability Model, in *Measurement Science and Technology* 2023. [[paper](https://iopscience.iop.org/article/10.1088/1361-6501/ad095a)]

* Unsupervised Statistical Feature-Guided Diffusion Model for Sensor-based Human Activity Recognition, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2306.05285)]


### Audio

* DiffWave: A Versatile Diffusion Model for Audio Synthesis, in *ICLR* 2021. [[paper](https://arxiv.org/abs/2009.09761)] [[official-code](https://diffwave-demo.github.io/)]

* Speech Enhancement and Dereverberation with Diffusion-based Generative Models, in *IEEE/ACM Transactions on Audio, Speech, and Language Processing* 2023. [[paper](https://arxiv.org/abs/2208.05830)] [[official-code](https://github.com/sp-uhh/sgmse)]

* StoRM: A Diffusion-based Stochastic Regeneration Model for Speech Enhancement and Dereverberation, *IEEE/ACM Transactions on Audio, Speech, and Language Processing* 2023. [[paper](https://arxiv.org/abs/2212.11851)] [[official-code](https://github.com/sp-uhh/storm)]

* Reducing the Prior Mismatch of Stochastic Differential Equations for Diffusion-based Speech Enhancement, in *Interspeech* 2023. [[paper](https://arxiv.org/abs/2302.14748)] [[official-code](https://github.com/sp-uhh/sgmse-bbed)]

* Fast Diffusion GAN Model for Symbolic Music Generation Controlled by Emotions, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2310.14040)]



## Related Diffusion Model Resources and Surveys

* A Survey on Generative Diffusion Models, in *arXiv* 2022. [[paper](https://arxiv.org/abs/2209.02646)] [[link](https://github.com/YangLing0818/Diffusion-Models-Papers-Survey-Taxonomy)]

* Diffusion Models: A Comprehensive Survey of Methods and Applications, in *ACM Computing Surveys* 2023. [[paper](https://arxiv.org/abs/2209.00796)] [[link](https://github.com/chq1155/A-Survey-on-Generative-Diffusion-Model)]

* Diffusion Models for Time Series Applications: A Survey, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2305.00624)]

* A Comprehensive Survey on Generative Diffusion Models for Structured Data, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2306.04139)] [[link](https://deepai.org/publication/a-survey-on-generative-diffusion-models-for-structured-data)]

* Awesome-Diffusion-Models. [[link](https://diff-usion.github.io/Awesome-Diffusion-Models/)]

