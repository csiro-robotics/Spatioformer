# Spatioformer
The official repository of "[Spatioformer: A Geo-encoded Transformer for Large-Scale Plant Species Richness Prediction](https://ieeexplore.ieee.org/abstract/document/10854505)" published at the IEEE Transactions on Geoscience and Remote Sensing, 2025. 

A preprint version of the paper can be found at https://arxiv.org/abs/2410.19256.

## Abstract

Earth observation data have shown promise in predicting species richness of vascular plants (α-diversity), but extending this approach to large spatial scales is challenging because geographically distant regions may exhibit different compositions of plant species (β-diversity), resulting in a location-dependent relationship between richness and spectral measurements. In order to handle such geolocation dependency, we propose Spatioformer, where a novel geolocation encoder is coupled with the transformer model to encode geolocation context into remote sensing imagery. The Spatioformer model compares favourably to state-of-the-art models in richness predictions on a large-scale ground-truth richness dataset (HAVPlot) that consists of 68,170 in-situ richness samples covering diverse landscapes across Australia. The results demonstrate that geolocational information is advantageous in predicting species richness from satellite observations over large spatial scales. With Spatioformer, plant species richness maps over Australia are compiled from Landsat archive for the years from 2015 to 2023. The richness maps produced in this study reveal the spatiotemporal dynamics of plant species richness in Australia, providing supporting evidence to inform effective planning and policy development for plant diversity conservation. Regions of high richness prediction uncertainties are identified, highlighting the need for future in-situ surveys to be conducted in these areas to enhance the prediction accuracy.

## News

[2024-12] The Spatioformer code released.


## Usage

- Create conda environment with python:

```
conda create --name spatioformer python=3.12.3
conda activate spatioformer
```

- Install required packages

```
pip install -r requirements.txt
```

- Download dataset

```
mkdir -p data_to_release
gdown --id 1GJvfBXmBEt9sH1I7wsvtR5WoYp1keyJz -O data_to_release/
gdown --id 1IHCzBvg6a-zEczo_gpgia9PiVLjD96XC -O data_to_release/
```

- Train the Spatioformer model

```
python run_spatioformer.py
```

The trained model will be saved at ./models/spatioformer/{time}/model.pth

## Reproducing Results

Please check out notebooks in the ./notebooks/ folder for reproducing figures and tables reported in the paper:

- ./notebooks/fig_geoencoding.ipynb for reproducing Figure 6;
- ./notebooks/fig_accuracy.ipynb for reproducing Figure 7 and Table 1;
- ./notebooks/fig_benchmark.ipynb for reproducing Figure 8;
- ./notebooks/fig_map.ipynb for reproducing Figures 9 and 10;
- ./notebooks/fig_uncertainty.ipynb for reporducing Figure 11;
- ./notebooks/tab_insize.ipynb for reproducing Table 2.

## Machine Learning-Ready Dataset

The plant species richness dataset of this study has been released in a machine learning-ready format at: https://data.csiro.au/collection/csiro:62308

## How to cite our paper

```
@article{guo2024spatioformer,
  title={Spatioformer: A Geo-encoded Transformer for Large-Scale Plant Species Richness Prediction},
  author={Guo, Yiqing and Mokany, Karel and Levick, Shaun R and Yang, Jinyan and Moghadam, Peyman},
  journal={arXiv preprint arXiv:2410.19256},
  year={2024}
}
```


## Acknowledgement

This study is supported by the Spatiotemporal Activity within CSIRO’s Machine Learning and Artificial Intelligence Future Science Platform, and the  Biodiversity Analytics From Space project within CSIRO’s Space Technology Future Science Platform.

The data sources of in-situ samples utilised in this study are cited in full in the Supporting information of Mokany et al. (2022), and include: data supplied by Department of Environment and Natural Resources © Northern Territory of Australia; Natural Values Atlas (www.naturalvaluesatlas.tas.gov.au), 2022, © State of Tasmania; NSW BioNet Flora Survey Data Collection © State Government of NSW and Department of Planning, Industry and Environment 2013; Queensland CORVEG Database, ver. 8/3/2019 State of Queensland (Department of Environment and Science, www.des.qld.gov.au/); Victorian Biodiversity Atlas © State Government of Victoria (accessed June 2017); NatureMap © State Government of Western Australia; NatureMaps © State Government of South Australia, Department for Environment and Water; TERN Ausplots, The Univ. of Adelaide (www.adelaide.edu.au), Adelaide, South Australia—supported by the Australian Government through the National Collaborative Research Infrastructure Strategy (NCRIS); Desert Ecology Research Group Plots © 2015–2018 Rights owned by the Univ. of Sydney; AusCover © 2011–2013 The Univ. of Queensland (Joint Remote Sensing Research Program).

Our thanks go to Dr Robert Woodcock, Mr Geoffrey Squire, and Mr Tisham Dhar at CSIRO for their invaluable advice on large-scale cloud computing, and Dr Simon Ferrier for his guidance on plant biodiversity analysis. We acknowledge the computational resources provided by the Earth Analytics Science and Innovation (EASI) platform and the high-performing computer Bracewell. We also acknowledge the satellite image archive provided by Geoscience
Australia’s Digital Earth Australia (DEA) program.

Reference: K. Mokany, J. K. McCarthy, D. S. Falster, R. V. Gallagher, T. D. Harwood, R. Kooyman, and M. Westoby, “Patterns and drivers of plant diversity across Australia,” Ecography, vol. 2022, no. 11, p. e06426, 2022.
