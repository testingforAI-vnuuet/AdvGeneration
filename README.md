# On Improvement of Adversarial Examples in Pattern-Based Autoencoder


## About the project
Contain source code of adversarial example generation methods, namely HPBA 

If you have any questions, please contact me via nguyenducanh@vnu.edu.vn.

## Table of Content 

* [About the Project](#about-the-project)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
  * [Using the pretrained model](#Using-the-pretrained-model)
  * [Dataset Preparation](#Dataset-Preparation)
    * [Training Data](#Training-data)
    * [Pre-trained DNN](#Pre-trained-DNN)
  * [View Result](#View-Result)
* [Run experiment](Run experiment)



## Getting started

### Prerequisites

* Python >= 3.7
* Tensorflow >= 2.4.0
### Installation
```sh
git clone https://github.com/testingforAI-vnuuet/AdvGeneration.git
cd AdvGeneration
git checkout papers/HPBA
pip install -r requirements.txt
```
### Dataset Preparation

#### Training data
- Use training set and label set after pro-processed (example: [mnist-training](https://drive.google.com/file/d/1R7gvFYTrtH75cV7qDg_zaQJ5J8ccIZCV/view?usp=sharing) & [mnist-label](https://drive.google.com/file/d/1miFdEi1X8Fr6hZx9_9UbWOmC8MPS27AJ/view?usp=sharing))
#### Pre-trained DNN
- Use pre-trained DNN that was training by chosen training data with high accuracy (example: [mnist-model](https://drive.google.com/file/d/1eBmWjM3HPp2Ci3e6dhd7iMNYCik2Se8q/view?usp=sharing))
### Input Configuration
- Input required configation in file [config.ini](config.ini)
### View Result
