# Diffusion Model for Time Series and Spatio-Temporal Data

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) 
![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green) 
![Stars](https://img.shields.io/github/stars/yyysjz1997/Awesome-TimeSeries-SpatioTemporal-Diffusion-Model)
[![Visits Badge](https://badges.pufler.dev/visits/yyysjz1997/Awesome-TimeSeries-SpatioTemporal-Diffusion-Model)](https://badges.pufler.dev/visits/yyysjz1997/Awesome-TimeSeries-SpatioTemporal-Diffusion-Model)


A curated list of **Diffusion Models** for **Time Series**, **SpatioTemporal Data** and **Tabular Data** with awesome resources (paper, code, application, review, survey, etc.), which aims to comprehensively and systematically summarize the recent advances to the best of our knowledge.

We will continue to update this list with the newest resources. If you find any missed resources (paper/code) or errors, please feel free to open an issue or make a pull request.

## Survey

<div align="center">
      <b>[<a href="https://arxiv.org/abs/2404.18886">✨ A Survey on Diffusion Models for Time Series and Spatio-Temporal Data</a>]</b>
</div>

<div align="center">

**[<a href="https://mp.weixin.qq.com/s/yUo6NDDeMRHAqoKk5RPZdg">专知 中文解读</a>]**

</div>


**Authors**: Yiyuan Yang, Ming Jin, Haomin Wen, Chaoli Zhang, Yuxuan Liang*, Lintao Ma, Yi Wang, Chenghao Liu, Bin Yang, Zenglin Xu, Jiang Bian, Shirui Pan, Qingsong Wen*

🙋 Please let us know if you find a mistake or have any suggestions!

✨ If you found the survey paper and this Repo useful, please consider to star this repository and cite our survey paper:

```bibtex
@article{yang2024survey,
      title={A Survey on Diffusion Models for Time Series and Spatio-Temporal Data}, 
      author={Yiyuan Yang and Ming Jin and Haomin Wen and Chaoli Zhang and Yuxuan Liang and Lintao Ma and Yi Wang and Chenghao Liu and Bin Yang and Zenglin Xu and Jiang Bian and Shirui Pan and Qingsong Wen},
      year={2024},
      eprint={2404.18886},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Taxonomy Document

✨ **[[Paper List and Taxonomy](https://docs.google.com/spreadsheets/d/1X5ujA-yXCzkzz4Uu9ErP8mu_Jg2zWk4fwZrwrMcqMoI/edit#gid=0)]**.

🙋 Please feel free to submit comments on any new works!



## Contents

- [Diffusion Model for Time Series and SpatioTemporal Data](#diffusion-model-for-time-series-and-spatiotemporal-data)
  - [Contents](#contents)
  - [Diffusion Model for Time Series Data](#diffusion-model-for-time-series-data)
    - [Prediction](#prediction)
    - [Generation](#generation)
    - [Imputation](#imputation)
    - [Anomaly Detection](#anomaly-detection)
    - [Classification and Regression](#classification-and-regression)
    - [Causal Inference](#causal-inference)
    - [Event Prediction and Classification](#event-prediction-and-classification)
  - [Diffusion Model for SpatioTemporal Data](#diffusion-model-for-spatiotemporal-data)
  - [Diffusion Model for Tabular Data](#diffusion-model-for-tabular-data)
  - [Applications](#applications)
    - [Healthcare](#healthcare)
    - [Sequential Recommendation](#sequential-recommendation)
    - [Weather](#weather)
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

### Prediction

* Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting, in *ICML* 2021. [[paper](https://arxiv.org/abs/2101.12072)] [[official-code](https://github.com/zalandoresearch/pytorch-ts)]

* ScoreGrad: Multivariate Probabilistic Time Series Forecasting with Continuous Energy-based Generative Models, in *arXiv* 2021. [[paper](https://arxiv.org/abs/2106.10121)] [[official-code](https://github.com/yantijin/ScoreGradPred)]

* Generative Time Series Forecasting with Diffusion, Denoise, and Disentanglement, in *NeurIPS* 2022. [[paper](https://arxiv.org/abs/2301.03028)] [[official-code](https://github.com/PaddlePaddle/PaddleSpatial/tree/main/research/D3VAE)]

* Diffusion Variational Autoencoder for Tackling Stochasticity in Multi-Step Regression Stock Price Prediction, in *CIKM* 2023. [[paper](https://arxiv.org/abs/2309.00073)] [[official-code](https://github.com/koa-fin/dva)]

* Modeling Temporal Data as Continuous Functions with Process Diffusion, in *ICML* 2023. [[paper](https://arxiv.org/abs/2211.02590)]

* Non-autoregressive Conditional Diffusion Models for Time Series Prediction, in *ICML* 2023. [[paper](https://dl.acm.org/doi/10.5555/3618408.3619692)]

* Graph Convolution Recurrent Denoising Diffusion Model for Multivariate Probabilistic Temporal Forecasting, in *International Conference on Advanced Data Mining and Applications (ADMA)* 2023. [[paper](https://dl.acm.org/doi/10.5555/3618408.3619692)]

* Data-driven and knowledge-guided denoising diffusion model for flood forecasting, in *Expert Systems with Applications* 2023. [[paper](https://www.sciencedirect.com/science/article/pii/S0957417423034103)]

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

* DiffDA: a diffusion model for weather-scale data assimilation, in *arXiv* 2024. [[paper](https://arxiv.org/abs/2401.05932)]

* A novel gear RUL prediction method by diffusion model generation health index and attention guided multi-hierarchy LSTM, in *Scientific Reports* 2024. [[paper](https://www.nature.com/articles/s41598-024-52151-y)]

* A transformer-based diffusion probabilistic model for heart rate and blood pressure forecasting in Intensive Care Unit, in *Computer Methods and Programs in Biomedicine* 2024. [[paper](https://www.sciencedirect.com/science/article/pii/S0169260724000567)] [[official-code](https://github.com/PingChang818/TDSTF)]

* Quantifying uncertainty: Air quality forecasting based on dynamic spatial-temporal denoising diffusion Probabilistic model, in *Environmental Research* 2024. [[paper](https://www.sciencedirect.com/science/article/pii/S0013935124003426)]

* DiffSTOCK: Probabilistic relational Stock Market Predictions using Diffusion Models, in *ICASSP* 2024. [[paper](https://arxiv.org/pdf/2403.14063)]

* Multi-Modality Conditional Diffusion Model for Time Series Forecasting of Live Sales Volume, in *ICASSP* 2024. [[paper](https://ieeexplore.ieee.org/abstract/document/10446307)]
  
* Latent Diffusion Transformer for Probabilistic Time Series Forecasting, in *AAAI* 2024. [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/29085)]
  
* DiffPLF: A Conditional Diffusion Model for Probabilistic Forecasting of EV Charging Load, in *arXiv* 2024. [[paper](https://arxiv.org/abs/2402.13548)] [[official-code](https://github.com/LSY-Cython/DiffPLF)]

* Comparative Analysis of TEC Prediction Models During Geomagnetic Storm and Quiet Conditions Using Diffusion, Transformer, and SARIMA, 2024. [[paper](https://essopenarchive.org/doi/full/10.22541/essoar.171291801.18571788/v1)]

* Residential Customer Baseline Load Estimation Based on Conditional Denoising Diffusion Probabilistic Model, in *IEEE 4th International Conference in Power Engineering Applications (ICPEA)* 2024. [[paper](https://ieeexplore.ieee.org/abstract/document/10498265)]


### Generation

* WaveGrad: Estimating Gradients for Waveform Generation, in *ICLR* 2021. [[paper](https://arxiv.org/abs/2009.00713)]

* DiffWave: A Versatile Diffusion Model for Audio Synthesis, in *ICLR* 2021. [[paper](https://arxiv.org/abs/2009.09761)] [[official-code](https://diffwave-demo.github.io/)]

* CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation, in *NeurIPS* 2021. [[paper](https://arxiv.org/pdf/2107.03502)] [[official-code](https://github.com/ermongroup/csdi)]

* Conditional Simulation Using Diffusion Schrödinger Bridges, in *UAI* 2022. [[paper](https://arxiv.org/abs/2202.13460)] [[official-code](https://github.com/vdeborto/cdsb)]

* Diffusing Gaussian Mixtures for Generating Categorical Data, in *AAAI* 2023. [[paper](https://arxiv.org/abs/2303.04635)] [[official-code](https://github.com/networkslab/gmcd)]

* Diffusion Generative Models in Infinite Dimensions, in *AISTATS* 2023. [[paper](https://arxiv.org/abs/2212.00886)] [[official-code](https://github.com/gavinkerrigan/functional_diffusion)]

* Multi-scale Conditional Diffusion Model for Deposited Droplet Volume Measurement in Inkjet Printing Manufacturing, in *Journal of Manufacturing Systems* 2023. [[paper](https://www.sciencedirect.com/science/article/pii/S0278612523002042)]
 
* Diffusion Augmentation for Sequential Recommendation, in *CIKM* 2023. [[paper](https://dl.acm.org/doi/abs/10.1145/3583780.3615134)] [[official-code](https://github.com/liuqidong07/DiffuASR)]

* Short-Term Wind Power Scenario Generation Based on Conditional Latent Diffusion Models, in *IEEE Transactions on Sustainable Energy* 2023. [[paper](https://ieeexplore.ieee.org/abstract/document/10296015)]

* Synthetic Sleep EEG Signal Generation using Latent Diffusion Models, in *DGM4H NeurIPS* 2023. [[paper](https://openreview.net/forum?id=mDwURmlapW)] [[official-code](https://github.com/bruAristimunha/Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models)]

* Generative AI Enables the Detection of Autism Using EEG Signals, in *Biometric Recognition* 2023. [[paper](https://link.springer.com/chapter/10.1007/978-981-99-8565-4_35)]

* TimeDDPM: Time series augmentation strategy for industrial soft sensing, in *IEEE Sensors Journal* 2023. [[paper](https://ieeexplore.ieee.org/abstract/document/10353969)]

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

* TS-Diffusion: Generating Highly Complex Time Series with Diffusion Models, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2311.03303)]

* Reliable Generation of EHR Time Series via Diffusion Models, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2310.15290v2)]

* D3A-TS: Denoising-Driven Data Augmentation in Time Series, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2312.05550)]

* Time Series Diffusion Method: A Denoising Diffusion Probabilistic Model for Vibration Signal Generation, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2312.07981)]

* Bayesian ECG reconstruction using denoising diffusion generative models, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2401.05388)]

* Regular Time-series Generation using SGM, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2301.08518)]

* On the Constrained Time-Series Generation Problem, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2307.01717)]

* A Short-Term Wind Power Scenario Generation Method Based on Conditional Diffusion Model, in *IEEE Sustainable Power and Energy Conference (iSPEC)* 2023. [[paper](https://ieeexplore.ieee.org/abstract/document/10403004)]

* Scenario Generation of Renewable Energy Based on Improved Diffusion Model, in *IEEE Sustainable Power and Energy Conference (iSPEC)* 2023. [[paper](https://ieeexplore.ieee.org/abstract/document/10402981)]

* Plug-in Diffusion Model for Sequential Recommendation, in *arXiv* 2024. [[paper](https://arxiv.org/pdf/2401.02913.pdf)][[official-code](https://github.com/hulkima/PDRec)]

* DiffuseRoll: multi-track multi-attribute music generation based on diffusion model, in *Multimedia Systems* 2024. [[paper](https://link.springer.com/article/10.1007/s00530-023-01220-9)] [[official-code](https://github.com/Fairywang9/DiffuseRoll)]
 
* BioDiffusion: A Versatile Diffusion Model for Biomedical Signal Synthesis, in *arXiv* 2024. [[paper](https://arxiv.org/abs/2401.10282)][[official-code](https://github.com/imics-lab/biodiffusion)]
 
* Risk-Sensitive Diffusion: Learning the Underlying Distribution from Noisy Samples, in *arXiv* 2024. [[paper](https://arxiv.org/abs/2402.02081)]

* Time Series Diffusion in the Frequency Domain, in *arXiv* 2024. [[paper](https://arxiv.org/abs/2402.05933)][[official-code](https://github.com/JonathanCrabbe/FourierDiffusion)]

* DiffsFormer: A Diffusion Transformer on Stock Factor Augmentation, in *arXiv* 2024. [[paper](https://arxiv.org/abs/2402.06656v1)]

* Time Weaver: A Conditional Time Series Generation Model, in *ArXiv* 2024. [[paper](https://arxiv.org/pdf/2403.02682v1)]

* Cold Diffusion Model for Seismic Denoising, 2024. [[paper](https://essopenarchive.org/users/750010/articles/721094-cold-diffusion-model-for-seismic-denoising)] [[official-code](https://github.com/Daniele-Trappolini/Diffusion-Model-for-Earthquake)]

* Conditional Diffusion Models as Self-supervised Learning Backbone for Irregular Time Series, in *ICLR Workshop on Learning from Time Series for Health* 2024. [[paper](https://scholar.google.com/scholar_url?url=https://openreview.net/pdf%3Fid%3DFay67M5R3R&hl=zh-CN&sa=X&d=3311036968068113244&ei=nV30ZZbIN7fHy9YPzam-kA8&scisig=AFWwaebu_vMvKQsDYVUKUZX9Li2u&oi=scholaralrt&hist=FUuGvZIAAAAJ:247144137884962227:AFWwaeY3zXyko7quKq0AnllX_8U0&html=&pos=0&folt=kw)]

* Synthesizing EEG Signals from Event-Related Potential Paradigms with Conditional Diffusion Models, in *ArXiv* 2024. [[paper](https://arxiv.org/pdf/2403.18486)]

* Creating synthetic energy meter data using conditional diffusion and building metadata, in *ArXiv* 2024. [[paper](https://arxiv.org/pdf/2404.00525v1)] [[official-code](https://github.com/buds-lab/energy-diffusion)]

* Balanced Mixed-Type Tabular Data Synthesis with Diffusion Models, in *ArXiv* 2024. [[paper](https://arxiv.org/pdf/2404.08254)]

* Long-form music generation with latent diffusion, in *ArXiv* 2024. [[paper](https://arxiv.org/pdf/2404.10301)] [[official-code](https://github.com/Stability-AI/stable-audio-tools/)]

* Cross-sensor super-resolution of irregularly sampled Sentinel-2 time series, in *ArXiv* 2024. [[paper](https://arxiv.org/abs/2404.16409)] [[official-code](https://github.com/aimiokab/MISR-S2)] 

  
### Imputation

* CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation, in *NeurIPS* 2021. [[paper](https://arxiv.org/abs/2107.03502)] [[official-code](https://github.com/ermongroup/CSDI)]
  
* Modeling Temporal Data as Continuous Functions with Process Diffusion, in *ICML* 2023. [[paper](https://arxiv.org/abs/2211.02590)]

* An Observed Value Consistent Diffusion Model for Imputing Missing Values in Multivariate Time Series, in *KDD* 2023. [[paper](https://dl.acm.org/doi/10.1145/3580305.3599257)]

* Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models, in *Transactions on Machine Learning Research (TMLR)* 2023. [[paper](https://arxiv.org/abs/2208.09399)] [[official-code](https://github.com/ai4healthuol/sssd)]

* Density-Aware Temporal Attentive Step-wise Diffusion Model For Medical Time Series Imputation, in *CIKM* 2023. [[paper](https://dl.acm.org/doi/abs/10.1145/3583780.3614840)]

* MEDiC: Mitigating EEG Data Scarcity Via Class-Conditioned Diffusion Model, in *DGM4H NeurIPS* 2023. [[paper](https://openreview.net/forum?id=0aeDKGhlTo)]

* Missing Data Filling in Soft Sensing Using Denoising Diffusion Probability Model, in *Measurement Science and Technology* 2023. [[paper](https://iopscience.iop.org/article/10.1088/1361-6501/ad095a)]

* Diffusion-Based Time Series Data Imputation for Cloud Failure Prediction at Microsoft 365, in *ESEC/FSE* 2023. [[paper](https://dl.acm.org/doi/abs/10.1145/3611643.3613866)]

* Sasdim: Self-adaptive Noise Scaling Diffusion Model for Spatial Time Series Imputation, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2309.01988)]

* Improving Diffusion Models for ECG Imputation with an Augmented Template Prior, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2310.15742)]

* Imputing time-series microbiome abundance profiles with diffusion model, in *IEEE International Conference on Bioinformatics and Biomedicine (BIBM)* 2023. [[paper](https://ieeexplore.ieee.org/abstract/document/10385703)] [[official-code](https://github.com/misatoseki/metag_time_impute)]

* SADI: Similarity-Aware Diffusion Model-Based Imputation for Incomplete Temporal EHR Data, in *International Conference on Artificial Intelligence and Statistics* 2024. [[paper](https://proceedings.mlr.press/v238/dai24c.html)]

* DiffsFormer: A Diffusion Transformer on Stock Factor Augmentation, in *arXiv* 2024. [[paper](https://arxiv.org/abs/2402.06656v1)]


### Anomaly Detection

* Imputation-based Time-Series Anomaly Detection with Conditional Weight-Incremental Diffusion Models, in *KDD* 2023. [[paper](https://dl.acm.org/doi/abs/10.1145/3580305.3599391)] [[official-code](https://github.com/ChunjingXiao/DiffAD)]

* Time Series Anomaly Detection using Diffusion-based Models, in *ICDM Workshop* 2023. [[paper](https://arxiv.org/abs/2311.01452)] [[official-code](https://github.com/fbrad/diffusionae)]

* Drift doesn’t Matter: Dynamic Decomposition with Diffusion Reconstruction for Unstable Multivariate Time Series Anomaly Detection, in *NeurIPS* 2023. [[paper](https://openreview.net/pdf?id=aW5bSuduF1)] [[official-code](https://github.com/ForestsKing/D3R)]
 
* Diff-SwinT: An Integrated Framework of Diffusion Model and Swin Transformer for Radar Jamming Recognition, in *Future Internet* 2023. [[paper](https://www.mdpi.com/1999-5903/15/12/374)]

* Diffusion-Based Time Series Data Imputation for Cloud Failure Prediction at Microsoft 365, in *ESEC/FSE* 2023. [[paper](https://dl.acm.org/doi/abs/10.1145/3611643.3613866)]

* ImDiffusion: Imputed Diffusion Models for Multivariate Time Series Anomaly Detection, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2307.00754)] [[official-code](https://github.com/17000cyh/IMDiffusion)]

* DDMT: Denoising Diffusion Mask Transformer Models for Multivariate Time Series Anomaly Detection, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2310.08800)] [[official-code](https://github.com/yangchaocheng/DDTM)]

* Generating HSR Bogie Vibration Signals via Pulse Voltage-Guided Conditional Diffusion Model, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2311.00496)] [[official-code](https://github.com/xuanliu2000/VGCDM)]

* Attention-enhanced conditional-diffusion-based data synthesis for data augmentation in machine fault diagnosis, in *Engineering Applications of Artificial Intelligence* 2024. [[paper](https://www.sciencedirect.com/science/article/pii/S0952197623018808)]

* DiffTAD: Denoising diffusion probabilistic models for vehicle trajectory anomaly detection, in *Knowledge-Based Systems* 2024. [[paper](https://www.sciencedirect.com/science/article/pii/S0950705124000224)] [[official-code](https://github.com/Psychic-DL/DiffTAD)]

* Diffusion-UDA: Diffusion-based unsupervised domain adaptation for submersible fault diagnosis, in *Electronics Letters* 2024. [[paper](https://ietresearch.onlinelibrary.wiley.com/doi/pdf/10.1049/ell2.13122)]

* Denoising diffusion implicit model for bearing fault diagnosis under different working loads, in *ITM Web of Conferences* 2024. [[paper](https://www.itm-conferences.org/articles/itmconf/abs/2024/06/itmconf_amict2023_01025/itmconf_amict2023_01025.html)]

* Unsupervised Anomaly Detection for Multivariate Time Series Using Diffusion Model in *ICASSP* 2024. [[paper](https://ieeexplore.ieee.org/abstract/document/10447083)] [[official-code](https://github.com/Hurongyao/TIMEADDM)]

* Anomaly Detection for Telemetry Time Series Using a Denoising Diffusion Probabilistic Model, in *IEEE sensors journal* 2024. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10494221)]

* A novel stochastic process diffusion model for wind turbines condition monitoring and fault identification with multi-parameter information fusion, in *Mechanical Systems and Signal Processing* 2024. [[paper](https://www.sciencedirect.com/science/article/pii/S0888327024002954#f0030)]


### Classification and Regression

* CARD: Classification and Regression Diffusion Models, in *NeurIPS* 2022. [[paper](https://arxiv.org/abs/2206.07275)] [[official-code](https://github.com/xzwhan/card)]

* Applying Generative Models and Transfer Learning to Physiological Data Classification, in *Artificial Intelligence Research and Development* 2023. [[paper](https://ebooks.iospress.nl/doi/10.3233/FAIA230656)]

* Generative AI Enables the Detection of Autism Using EEG Signals, in *Biometric Recognition* 2023. [[paper](https://link.springer.com/chapter/10.1007/978-981-99-8565-4_35)]

* Enhanced Atrial Fibrillation (AF) Detection via Data Augmentation with Diffusion Model, in *International Conference on Computer and Knowledge Engineering* 2023. [[paper](https://ieeexplore.ieee.org/abstract/document/10326310)] [[official-code](https://github.com/arashVsh/AF-Detection-DDPM)]

* Brain-Driven Representation Learning Based on Diffusion Model, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2311.07925)]

* AIGC for RF Sensing: The Case of RFID-based Human Activity Recognition, in *International Conference on Computing, Networking and Communications* 2024. [[paper](http://www.conf-icnc.org/2024/papers/p1092-wang.pdf)]

* ECG Captioning with Prior-Knowledge Transformer and Diffusion Probabilistic Model, in *Journal of Healthcare Informatics Research* 2024. [[paper](https://www.researchsquare.com/article/rs-4070328/v1)]

* Diffusion Language-Shapelets for Semi-supervised Time-Series Classification, in *AAAI* 2024. [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/29317)] [[official-code](https://github.com/qianlima-lab/DiffShape)]

* Evaluating Diffusion Models for the Automation of Ultrasonic Nondestructive Evaluation Data Analysis, in *Algorithms* 2024. [[paper](https://www.mdpi.com/1999-4893/17/4/167)]


### Causal Inference

* Diffusion Model in Causal Inference with Unmeasured Confounders, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2308.03669)] [[official-code](https://github.com/tatsu432/BDCM)]

* Brain Diffuser with Hierarchical Transformer for MCI Causality Analysis, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2312.09022)]


### Event Prediction and Classification

* Non-Autoregressive Diffusion-based Temporal Point Processes for Continuous-Time Long-Term Event Prediction, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2311.01033)]

* Add and Thin: Diffusion for Temporal Point Processes, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2311.01139)] [[official-code](https://www.cs.cit.tum.de/daml/add-thin/)]

* Adversarial Purification for Data-Driven Power System Event Classifiers with Diffusion Models, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2311.07110)]


## Diffusion Model for SpatioTemporal Data

* Spatio-temporal Diffusion Point Processes, in *KDD* 2023. [[paper](https://arxiv.org/abs/2305.12403)] [[official-code](https://github.com/tsinghua-fib-lab/Spatio-temporal-Diffusion-Point-Processes)]

* DYffusion: A Dynamics-informed Diffusion Model for Spatiotemporal Forecasting, in *NeurIPS* 2023. [[paper](https://arxiv.org/abs/2306.01984)] [[official-code](https://github.com/Rose-STL-Lab/dyffusion)]

* Extraction and Recovery of Spatio-Temporal Structure in Latent Dynamics Alignment with Diffusion Model, in *NeurIPS* 2023. [[paper](https://arxiv.org/abs/2306.06138)] [[official-code](https://github.com/alexwangntl/erdiff)]

* DiffSTG: Probabilistic Spatio-Temporal Graph with Denoising Diffusion Models, in *SIGSPATIAL* 2023. [[paper](https://arxiv.org/abs/2301.13629)]

* PriSTI: A Conditional Diffusion Framework for Spatiotemporal Imputation, in *ICDE* 2023. [[paper](https://arxiv.org/abs/2302.09746)] [[official-code](https://github.com/lmzzml/pristi)]

* DiffUFlow: Robust Fine-grained Urban Flow Inference with Denoising Diffusion Model, in *CIKM* 2023. [[paper](https://dl.acm.org/doi/abs/10.1145/3583780.3614842)]

* DiffTraj: Generating GPS Trajectory with Diffusion Probabilistic Model, in *NeurIPS* 2023. [[paper](https://arxiv.org/abs/2304.11582)] [[official-code](https://github.com/Yasoz/DiffTraj)]

* ControlTraj: Controllable Trajectory Generation with Topology-Constrained Diffusion Model, in *arXiv* 2024. [[paper](https://arxiv.org/pdf/2404.15380)]  [[official-code](https://github.com/aimiokab/MISR-S2)]
  
* Diff-RNTraj: A Structure-aware Diffusion Model for Road Network-constrained Trajectory Generation, in *arXiv* 2024. [[paper](https://arxiv.org/pdf/2402.07369)]

* Origin-Destination Travel Time Oracle for Map-based Services, in *SIGMOD* 2023. [[paper](https://arxiv.org/abs/2307.03048)]

* A Graph-Based Scene Encoder for Vehicle Trajectory Prediction Using the Diffusion Model, in *CSIS-IAC* 2023. [[paper](https://ieeexplore.ieee.org/abstract/document/10363970)]

* Dynamic Causal Explanation Based Diffusion-Variational Graph Neural Network for Spatio-temporal Forecasting, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2305.09703)] [[official-code](https://github.com/gorgen2020/dvgnn)]

* Sasdim: Self-adaptive Noise Scaling Diffusion Model for Spatial Time Series Imputation, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2309.01988)]

* Imputation as Inpainting: Diffusion Models for Spatiotemporal Data Imputation, in *OpenReview* 2023. [[paper](https://openreview.net/forum?id=QUANtQnx30l)]

* Towards Unifying Diffusion Models for Probabilistic Spatio-Temporal Graph Learning, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2310.17360)]

* Predicting the Position Uncertainty at the Time of Closest Approach with Diffusion Models, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2311.05417)]

* ChatTraffic: Text-to-Traffic Generation via Diffusion Model, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2311.16203)] [[official-code](https://github.com/ChyaZhang/ChatTraffic)]

* SpecSTG: A Fast Spectral Diffusion Framework for Probabilistic Spatio-Temporal Traffic Forecasting, in *arXiv* 2024. [[paper](https://arxiv.org/abs/2401.08119)] [[official-code](https://anonymous.4open.science/r/SpecSTG/)]

* DiffTAD: Denoising diffusion probabilistic models for vehicle trajectory anomaly detection, in *Knowledge-Based Systems* 2024. [[paper](https://www.sciencedirect.com/science/article/pii/S0950705124000224)] [[official-code](https://github.com/Psychic-DL/DiffTAD)]

* Simulating human mobility with a trajectory generation framework based on diffusion model, in *International Journal of Geographical Information Science* 2024. [[paper](https://www.tandfonline.com/doi/full/10.1080/13658816.2024.2312199)] [[official-code](https://github.com/chuchen2017/TrajGDM)]

* Learning Autoencoder Diffusion Models of Pedestrian Group Relationships for Multimodal Trajectory Prediction, in *IEEE Transactions on Instrumentation and Measurement* 2024. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10466609)]
  
* SRNDiff: Short-term Rainfall Nowcasting with Condition Diffusion Model, in *arXiv* 2024. [[paper](https://arxiv.org/abs/2402.13737)]

* Intention-aware Denoising Diffusion Model for Trajectory Prediction, in *ArXiv* 2024. [[paper](https://arxiv.org/pdf/2403.09190)]

* Graph-Jigsaw Conditioned Diffusion Model for Skeleton-based Video Anomaly Detection, in *ArXiv* 2024. [[paper](https://arxiv.org/pdf/2403.12172)]

* Diffusion-Based Environment-Aware Trajectory Prediction, in *ArXiv* 2024. [[paper](https://arxiv.org/pdf/2403.11643)]

* Diff-RNTraj: A Structure-aware Diffusion Model for Road Network-constrained Trajectory Generation, in *ArXiv* 2024. [[paper](https://www.arxiv.org/pdf/2402.07369)]

* Utilizing a Diffusion Model for Pedestrian Trajectory Prediction in Semi-Open Autonomous Driving Environments, in *IEEE sensors journal* 2024. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10489838)]

* Fault Detection in Mobile Networks Using Diffusion Models, in *ArXiv* 2024. [[paper](https://arxiv.org/pdf/2404.09240)]

* Graphusion: Latent Diffusion for Graph Generation, in *TKDE* 2024. [[paper](https://www.computer.org/csdl/journal/tk/5555/01/10508504/1Wr262l3Gg0)]

  
## Diffusion Model for Tabular Data

* Diffusion Models for Missing Value Imputation in Tabular Data, in *NeurIPS TRL Workshop* 2022. [[paper](https://arxiv.org/abs/2210.17128)] [[official-code](https://github.com/pfnet-research/TabCSDI)]

* MissDiff: Training Diffusion Models on Tabular Data with Missing Values, in *ICML Workshop* 2023. [[paper](https://arxiv.org/abs/2307.00467)]

* CoDi: Co-evolving Contrastive Diffusion Models for Mixed-type Tabular Synthesis, in *ICML* 2023. [[paper](https://arxiv.org/abs/2304.12654)] [[official-code](https://github.com/chaejeonglee/codi)]

* TabDDPM: Modelling Tabular Data with Diffusion Models, in *ICML* 2023. [[paper](https://arxiv.org/abs/2209.15421)] [[official-code](https://github.com/yandex-research/tab-ddpm)]

* Multi-task oriented diffusion model for mortality prediction in shock patients with incomplete data, in *Information Fusion* 2023. [[paper](https://www.sciencedirect.com/science/article/pii/S1566253523005237)]
 
* Conditioning Score-Based Generative Models by Neuro-Symbolic Constraints, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2308.16534)]

* On Diffusion Modeling for Anomaly Detection, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2305.18593)] [[official-code](https://github.com/vicliv/dte)]

* Generating Tabular Datasets under Differential Privacy, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2308.14784)]

* TabADM: Unsupervised Tabular Anomaly Detection with Diffusion Models, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2307.12336)]

* FinDiff: Diffusion Models for Financial Tabular Data Generation, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2309.01472)]

* AutoDiff: combining Auto-encoder and Diffusion model for tabular data synthesizing, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2310.15479)]

* DiffImpute: Tabular Data Imputation With Denoising Diffusion Probabilistic Model in *ArXiv* 2024. [[paper](https://arxiv.org/pdf/2403.13863)] [[official-code](https://github.com/dendiiiii/diffimpute)]
  
* RF-Diffusion: Radio Signal Generation via Time-Frequency Diffusion, in *MobiCom* 2024. [[paper](https://arxiv.org/pdf/2404.09140)] [[official-code](https://github.com/mobicom24/RF-Diffusion)]

* Guided Discrete Diffusion for Electronic Health Record Generation, in *ArXiv* 2024. [[paper](https://arxiv.org/pdf/2404.12314)]


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

* Data Augmentation for EEG Motor Imagery Classification Using Diffusion Model, in *Data Science and Artificial Intelligence conference* 2023. [[paper](https://link.springer.com/chapter/10.1007/978-981-99-7969-1_9)]

* Generative AI Enables the Detection of Autism Using EEG Signals, in *Biometric Recognition* 2023. [[paper](https://link.springer.com/chapter/10.1007/978-981-99-8565-4_35)]

* Enhanced Atrial Fibrillation (AF) Detection via Data Augmentation with Diffusion Model, in *International Conference on Computer and Knowledge Engineering* 2023. [[paper](https://ieeexplore.ieee.org/abstract/document/10326310)] [[official-code](https://github.com/arashVsh/AF-Detection-DDPM)]

* Multi-task oriented diffusion model for mortality prediction in shock patients with incomplete data, in *Information Fusion* 2023. [[paper](https://www.sciencedirect.com/science/article/pii/S1566253523005237)]

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

* Brain-Driven Representation Learning Based on Diffusion Model, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2311.07925)]

* Reliable Generation of EHR Time Series via Diffusion Models, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2310.15290v2)]

* Brain Diffuser with Hierarchical Transformer for MCI Causality Analysis, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2312.09022)]

* Bayesian ECG reconstruction using denoising diffusion generative models, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2401.05388)]

* Imputing time-series microbiome abundance profiles with diffusion model, in *IEEE International Conference on Bioinformatics and Biomedicine (BIBM)* 2023. [[paper](https://ieeexplore.ieee.org/abstract/document/10385703)] [[official-code](https://github.com/misatoseki/metag_time_impute)]

* Addiction-related brain networks identification via Graph Diffusion Reconstruction Network, in *Brain Informatics* 2024. [[paper](https://link.springer.com/article/10.1186/s40708-023-00216-5)]

* BioDiffusion: A Versatile Diffusion Model for Biomedical Signal Synthesis, in *arXiv* 2024. [[paper](https://arxiv.org/abs/2401.10282)][[official-code](https://github.com/imics-lab/biodiffusion)]

* A transformer-based diffusion probabilistic model for heart rate and blood pressure forecasting in Intensive Care Unit, in *Computer Methods and Programs in Biomedicine* 2024. [[paper](https://www.sciencedirect.com/science/article/pii/S0169260724000567)] [[official-code](https://github.com/PingChang818/TDSTF)]

* ECG Captioning with Prior-Knowledge Transformer and Diffusion Probabilistic Model, in *Journal of Healthcare Informatics Research* 2024. [[paper](https://www.researchsquare.com/article/rs-4070328/v1)]

* Synthesizing EEG Signals from Event-Related Potential Paradigms with Conditional Diffusion Models, in *ArXiv* 2024. [[paper](https://arxiv.org/pdf/2403.18486)]

* SADI: Similarity-Aware Diffusion Model-Based Imputation for Incomplete Temporal EHR Data, in *International Conference on Artificial Intelligence and Statistics* 2024. [[paper](https://proceedings.mlr.press/v238/dai24c.html)]

* Guided Discrete Diffusion for Electronic Health Record Generation, in *ArXiv* 2024. [[paper](https://arxiv.org/pdf/2404.12314)]
  
  
### Sequential Recommendation

* Recommendation via Collaborative Diffusion Generative Model, in *Knowledge Science, Engineering and Management: 15th International Conference (KSEM)* 2022. [[paper](https://dl.acm.org/doi/10.1007/978-3-031-10989-8_47)]

* Diffusion Recommender Model, in *SIGIR* 2023. [[paper](https://arxiv.org/abs/2304.04971)] [[official-code](https://github.com/yiyanxu/diffrec)]

* Diffusion Augmentation for Sequential Recommendation, in *CIKM* 2023. [[paper](https://dl.acm.org/doi/abs/10.1145/3583780.3615134)] [[official-code](https://github.com/liuqidong07/DiffuASR)]

* Discrete Conditional Diffusion for Reranking in Recommendation, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2308.06982)]

* RecFusion: A Binomial Diffusion Process for 1D Data for Recommendation, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2306.08947)] [[official-code](https://github.com/gabriben/recfusion)]

* Conditional Denoising Diffusion for Sequential Recommendation, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2304.11433)]

* Sequential Recommendation with Diffusion Models, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2304.04541)]

* DiffuRec: A Diffusion Model for Sequential Recommendation, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2304.00686)]

* Towards Graph-Aware Diffusion Modeling for Collaborative Filtering, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2311.08744)]

* Generate What You Prefer: Reshaping Sequential Recommendation via Guided Diffusion, in *NeurIPS* 2024. [[paper](https://arxiv.org/abs/2310.20453)] [[official-code](https://github.com/yangzhengyi98/dreamrec)]

* Plug-in Diffusion Model for Sequential Recommendation, in *arXiv* 2024. [[paper](https://arxiv.org/pdf/2401.02913.pdf)][[official-code](https://github.com/hulkima/PDRec)]


### Weather

* SwinRDM: Integrate SwinRNN with Diffusion Model towards High-Resolution and High-Quality Weather Forecasting, in *AAAI* 2023. [[paper](https://arxiv.org/abs/2306.03110)]

* DiffESM: Conditional Emulation of Earth System Models with Diffusion Models, in *ICLR* 2023. [[paper](https://arxiv.org/abs/2304.11699)] [[official-code](https://github.com/duanyiqun/ds-ddpm)]

* Precipitation Nowcasting with Generative Diffusion Models, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2308.06733)] [[official-code](https://github.com/fmerizzi/precipitation-nowcasting-with-generative-diffusion-models)]

* SEEDS: Emulation of Weather Forecast Ensembles with Diffusion Models, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2306.14066)]

* Diffusion Models for High-Resolution Solar Forecasts, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2302.00170)]

* Latent Diffusion Models for Generative Precipitation Nowcasting with Accurate Uncertainty Quantification, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2304.12891)] [[official-code](https://github.com/MeteoSwiss/ldcast)]

* WiREDiff: a Wind Resolution-Enhancing Diffusion Model, in *Working Paper* 2023. [[link](https://rupakv.com/wirediff.html#:~:text=WiREDiff%20is%20a%20novel%20application,interval%20of%20predicted%20wind%20velocities.)] [[paper](https://rupakv.com/pdfs/wirediff_report.pdf)] [[official-code](https://github.com/RupaKurinchiVendhan/WiREDiff)]

* DiffDA: a diffusion model for weather-scale data assimilation, in *arXiv* 2024. [[paper](https://arxiv.org/abs/2401.05932)]

* SRNDiff: Short-term Rainfall Nowcasting with Condition Diffusion Model, in *arXiv* 2024. [[paper](https://arxiv.org/abs/2402.13737)]


### Energy and Electricity

* Short-Term Wind Power Scenario Generation Based on Conditional Latent Diffusion Models, in *IEEE Transactions on Sustainable Energy* 2023. [[paper](https://ieeexplore.ieee.org/abstract/document/10296015)]

* Diff-SwinT: An Integrated Framework of Diffusion Model and Swin Transformer for Radar Jamming Recognition, in *Future Internet* 2023. [[paper](https://www.mdpi.com/1999-5903/15/12/374)]

* DiffCharge: Generating EV Charging Scenarios via a Denoising Diffusion Model, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2308.09857)] [[official-code](https://github.com/LSY-Cython/DiffCharge)]

* Customized Load Profiles Synthesis for Electricity Customers Based on Conditional Diffusion Models, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2304.12076)]

* DiffLoad: Uncertainty Quantification in Load Forecasting with Diffusion Model, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2306.01001)]

* Denoising Diffusion Probabilistic Models for Probabilistic Energy Forecasting, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2212.02977)] [[official-code](https://github.com/EstebanHernandezCapel/DDPM-Power-systems-forecasting)]

* Adversarial Purification for Data-Driven Power System Event Classifiers with Diffusion Models, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2311.07110)]

* Improved Efficient Two-Stage Denoising Diffusion Power System Measurement Recovery Against False Data Injection Attacks and Data Losses, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2312.04346)]

* A Short-Term Wind Power Scenario Generation Method Based on Conditional Diffusion Model, in *IEEE Sustainable Power and Energy Conference (iSPEC)* 2023. [[paper](https://ieeexplore.ieee.org/abstract/document/10403004)]

* Scenario Generation of Renewable Energy Based on Improved Diffusion Model, in *IEEE Sustainable Power and Energy Conference (iSPEC)* 2023. [[paper](https://ieeexplore.ieee.org/abstract/document/10402981)]

* Residential Customer Baseline Load Estimation Based on Conditional Denoising Diffusion Probabilistic Model, in *IEEE 4th International Conference in Power Engineering Applications (ICPEA)* 2024. [[paper](https://ieeexplore.ieee.org/abstract/document/10498265)]

* DiffPLF: A Conditional Diffusion Model for Probabilistic Forecasting of EV Charging Load, in *arXiv* 2024. [[paper](https://arxiv.org/abs/2402.13548)] [[official-code](https://github.com/LSY-Cython/DiffPLF)]

* Creating synthetic energy meter data using conditional diffusion and building metadata, in *ArXiv* 2024. [[paper](https://arxiv.org/pdf/2404.00525v1)] [[official-code](https://github.com/buds-lab/energy-diffusion)]


### Math and Physics

* A Physics-informed Diffusion Model for High-fidelity Flow Field Reconstruction, in *Journal of Computational Physics* 2023. [[paper](https://arxiv.org/abs/2211.14680)] [[official-code](https://github.com/BaratiLab/Diffusion-based-Fluid-Super-resolution)]

* DiTTO: Diffusion-inspired Temporal Transformer Operator, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2307.09072)]

* Infinite-dimensional Diffusion Models for Function Spaces, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2302.10130)]

* Generative Diffusion Learning for Parametric Partial Differential Equations, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2305.14703)]


### Finance

* Diffusion Variational Autoencoder for Tackling Stochasticity in Multi-Step Regression Stock Price Prediction, in *CIKM* 2023. [[paper](https://arxiv.org/abs/2309.00073)] [[official-code](https://github.com/koa-fin/dva)]

* FinDiff: Diffusion Models for Financial Tabular Data Generation, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2309.01472)]

* Synthetic Data Applications in Finance, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2401.00081)]

* DiffSTOCK: Probabilistic relational Stock Market Predictions using Diffusion Models, in *ICASSP* 2024. [[paper](https://arxiv.org/pdf/2403.14063)]

* Multi-Modality Conditional Diffusion Model for Time Series Forecasting of Live Sales Volume, in *ICASSP* 2024. [[paper](https://ieeexplore.ieee.org/abstract/document/10446307)]

* DiffsFormer: A Diffusion Transformer on Stock Factor Augmentation, in *arXiv* 2024. [[paper](https://arxiv.org/abs/2402.06656v1)]


### AIOps

* Maat: Performance Metric Anomaly Anticipation for Cloud Services with Conditional Diffusion, in *IEEE/ACM International Conference on Automated Software Engineering* 2023. [[paper](https://arxiv.org/abs/2308.07676)] [[official-code](https://github.com/bebillionaireusd/maat)]

* Diffusion-Based Time Series Data Imputation for Cloud Failure Prediction at Microsoft 365, in *ESEC/FSE* 2023. [[paper](https://dl.acm.org/doi/abs/10.1145/3611643.3613866)]

* NetDiffus: Network Traffic Generation by Diffusion Models through Time-Series Imaging, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2310.04429)]


### Environment

* TimeDDPM: Time series augmentation strategy for industrial soft sensing, in *IEEE Sensors Journal* 2023. [[paper](https://ieeexplore.ieee.org/abstract/document/10353969)]

* Deep Diffusion Models for Seismic Processing, in *Computers & Geosciences* 2023. [[paper](https://arxiv.org/abs/2207.10451)]

* Data-driven and knowledge-guided denoising diffusion model for flood forecasting, in *Expert Systems with Applications* 2023. [[paper](https://www.sciencedirect.com/science/article/pii/S0957417423034103)]

* Time Series Diffusion Method: A Denoising Diffusion Probabilistic Model for Vibration Signal Generation, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2312.07981)]

* Quantifying uncertainty: Air quality forecasting based on dynamic spatial-temporal denoising diffusion Probabilistic model, in *Environmental Research* 2024. [[paper](https://www.sciencedirect.com/science/article/pii/S0013935124003426)]

* Exploring denoising diffusion probabilistic model for daily streamflow gap filling in Central Asia typical watersheds, in *Journal of Hydrology: Regional Studies* 2024. [[paper](https://www.sciencedirect.com/science/article/pii/S2214581824000491)]

* Cold Diffusion Model for Seismic Denoising, 2024. [[paper](https://essopenarchive.org/users/750010/articles/721094-cold-diffusion-model-for-seismic-denoising)] [[official-code](https://github.com/Daniele-Trappolini/Diffusion-Model-for-Earthquake)]

* Comparative Analysis of TEC Prediction Models During Geomagnetic Storm and Quiet Conditions Using Diffusion, Transformer, and SARIMA, 2024. [[paper](https://essopenarchive.org/doi/full/10.22541/essoar.171291801.18571788/v1)]  


### Traffic

* DiffUFlow: Robust Fine-grained Urban Flow Inference with Denoising Diffusion Model, in *CIKM* 2023. [[paper](https://dl.acm.org/doi/abs/10.1145/3583780.3614842)]

* Origin-Destination Travel Time Oracle for Map-based Services, in *SIGMOD* 2023. [[paper](https://arxiv.org/abs/2307.03048)]

* A Graph-Based Scene Encoder for Vehicle Trajectory Prediction Using the Diffusion Model, in *CSIS-IAC* 2023. [[paper](https://ieeexplore.ieee.org/abstract/document/10363970)]

* Generating HSR Bogie Vibration Signals via Pulse Voltage-Guided Conditional Diffusion Model, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2311.00496)] [[official-code](https://github.com/xuanliu2000/VGCDM)]

* SpecSTG: A Fast Spectral Diffusion Framework for Probabilistic Spatio-Temporal Traffic Forecasting, in *arXiv* 2024. [[paper](https://arxiv.org/abs/2401.08119)] [[official-code](https://anonymous.4open.science/r/SpecSTG/)]

* DiffTAD: Denoising diffusion probabilistic models for vehicle trajectory anomaly detection, in *Knowledge-Based Systems* 2024. [[paper](https://www.sciencedirect.com/science/article/pii/S0950705124000224)] [[official-code](https://github.com/Psychic-DL/DiffTAD)]

* Intention-aware Denoising Diffusion Model for Trajectory Prediction, in *ArXiv* 2024. [[paper](https://arxiv.org/pdf/2403.09190)]

* Diffusion-Based Environment-Aware Trajectory Prediction, in *ArXiv* 2024. [[paper](https://arxiv.org/pdf/2403.11643)]

* Diff-RNTraj: A Structure-aware Diffusion Model for Road Network-constrained Trajectory Generation, in *ArXiv* 2024. [[paper](https://www.arxiv.org/pdf/2402.07369)]

* Utilizing a Diffusion Model for Pedestrian Trajectory Prediction in Semi-Open Autonomous Driving Environments, in *IEEE sensors journal* 2024. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10489838)]



### Manufacturing

* Multi-scale Conditional Diffusion Model for Deposited Droplet Volume Measurement in Inkjet Printing Manufacturing, in *Journal of Manufacturing Systems* 2023. [[paper](https://www.sciencedirect.com/science/article/pii/S0278612523002042)]

* Missing Data Filling in Soft Sensing Using Denoising Diffusion Probability Model, in *Measurement Science and Technology* 2023. [[paper](https://iopscience.iop.org/article/10.1088/1361-6501/ad095a)]

* Unsupervised Statistical Feature-Guided Diffusion Model for Sensor-based Human Activity Recognition, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2306.05285)]

* Attention-enhanced conditional-diffusion-based data synthesis for data augmentation in machine fault diagnosis, in *Engineering Applications of Artificial Intelligence* 2024. [[paper](https://www.sciencedirect.com/science/article/pii/S0952197623018808)]

* A novel gear RUL prediction method by diffusion model generation health index and attention guided multi-hierarchy LSTM, in *Scientific Reports* 2024. [[paper](https://www.nature.com/articles/s41598-024-52151-y)]

* Diffusion-UDA: Diffusion-based unsupervised domain adaptation for submersible fault diagnosis, in *Electronics Letters* 2024. [[paper](https://ietresearch.onlinelibrary.wiley.com/doi/pdf/10.1049/ell2.13122)]

* Denoising diffusion implicit model for bearing fault diagnosis under different working loads, in *ITM Web of Conferences* 2024. [[paper](https://www.itm-conferences.org/articles/itmconf/abs/2024/06/itmconf_amict2023_01025/itmconf_amict2023_01025.html)]

* AIGC for RF Sensing: The Case of RFID-based Human Activity Recognition, in *International Conference on Computing, Networking and Communications* 2024. [[paper](http://www.conf-icnc.org/2024/papers/p1092-wang.pdf)]

* Online prediction of mechanical and electrical quality in ultrasonic metal welding using time series generation and deep learning, in *Engineering Failure Analysis* 2024. [[paper](https://www.sciencedirect.com/science/article/pii/S1350630724002085)]

* A novel stochastic process diffusion model for wind turbines condition monitoring and fault identification with multi-parameter information fusion, in *Mechanical Systems and Signal Processing* 2024. [[paper](https://www.sciencedirect.com/science/article/pii/S0888327024002954#f0030)]
  

### Audio

* DiffWave: A Versatile Diffusion Model for Audio Synthesis, in *ICLR* 2021. [[paper](https://arxiv.org/abs/2009.09761)] [[official-code](https://diffwave-demo.github.io/)]

* Speech Enhancement with Score-Based Generative Models in the Complex STFT Domain, in *Interspeech* 2022. [[paper](https://arxiv.org/pdf/2203.17004)]

* Conditional Diffusion Probabilistic Model for Speech Enhancement in *ICASSP* 2022. [[paper](https://arxiv.org/pdf/2202.05256)] [[official-code](https://github.com/neillu23/CDiffuSE)]

* Universal Speech Enhancement with Score-based Diffusion in *ArXiv* 2022. [[paper](https://arxiv.org/pdf/2206.03065)]
  
* Speech Enhancement and Dereverberation with Diffusion-based Generative Models, in *IEEE/ACM Transactions on Audio, Speech, and Language Processing* 2023. [[paper](https://arxiv.org/abs/2208.05830)] [[official-code](https://github.com/sp-uhh/sgmse)]

* StoRM: A Diffusion-based Stochastic Regeneration Model for Speech Enhancement and Dereverberation, *IEEE/ACM Transactions on Audio, Speech, and Language Processing* 2023. [[paper](https://arxiv.org/abs/2212.11851)] [[official-code](https://github.com/sp-uhh/storm)]

* Reducing the Prior Mismatch of Stochastic Differential Equations for Diffusion-based Speech Enhancement, in *Interspeech* 2023. [[paper](https://arxiv.org/abs/2302.14748)] [[official-code](https://github.com/sp-uhh/sgmse-bbed)]

* Revisiting denoising diffusion probabilistic models for speech enhancement: Condition collapse, efficiency and refinement, in *AAAI* 2023. [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/26597)] [[official-code](https://github.com/judiebig/DR-DiffuSE)]

* Dose: Diffusion dropout with adaptive prior for speech enhancement, in *NeurIPS* 2023. [[paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/7e966a12c2d6307adb8809aaa9acf057-Abstract-Conference.html)] [[official-code](https://github.com/ICDM-UESTC/DOSE)]

* A Study on Speech Enhancement Based on Diffusion Probabilistic Model, in *APSIPA* 2023. [[paper](https://arxiv.org/pdf/2107.11876)] [[official-code](https://github.com/neillu23/DiffuSE)]

* CRA-DIFFUSE: Improved Cross-Domain Speech Enhancement Based on Diffusion Model with T-F Domain Pre-Denoising, in *ICME* 2023. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10219860)]

* Fast Diffusion GAN Model for Symbolic Music Generation Controlled by Emotions, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2310.14040)]

* DiffuseRoll: multi-track multi-attribute music generation based on diffusion model, in *Multimedia Systems* 2024. [[paper](https://link.springer.com/article/10.1007/s00530-023-01220-9)] [[official-code](https://github.com/Fairywang9/DiffuseRoll)]

* Long-form music generation with latent diffusion, in *ArXiv* 2024. [[paper](https://arxiv.org/pdf/2404.10301)] [[official-code](https://github.com/Stability-AI/stable-audio-tools/)]


## Related Diffusion Model Resources and Surveys

* A Survey on Generative Diffusion Models, in *arXiv* 2022. [[paper](https://arxiv.org/abs/2209.02646)] [[link](https://github.com/YangLing0818/Diffusion-Models-Papers-Survey-Taxonomy)]

* Diffusion Models: A Comprehensive Survey of Methods and Applications, in *ACM Computing Surveys* 2023. [[paper](https://arxiv.org/abs/2209.00796)] [[link](https://github.com/chq1155/A-Survey-on-Generative-Diffusion-Model)]

* Diffusion Models for Time Series Applications: A Survey, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2305.00624)]

* A Comprehensive Survey on Generative Diffusion Models for Structured Data, in *arXiv* 2023. [[paper](https://arxiv.org/abs/2306.04139)] [[link](https://deepai.org/publication/a-survey-on-generative-diffusion-models-for-structured-data)]

* Awesome-Diffusion-Models. [[link](https://diff-usion.github.io/Awesome-Diffusion-Models/)]

