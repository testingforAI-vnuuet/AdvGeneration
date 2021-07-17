# On Improvement of Adversarial Examples in Pattern-Based Autoencoder


## About the project
Contain source code of adversarial example generation method, namely HPBA 

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

* [Run experiment](#Run-experiment)
  * [Run](#run)
  * [View Result](#View-Result)


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
### Data Preparation

#### Training data
- Use training set and label set after being pre-processed. Example:
  - Hand-written digit MNIST: 
      - training data:[handwritten_mnist_training](https://drive.google.com/file/d/1R7gvFYTrtH75cV7qDg_zaQJ5J8ccIZCV/view?usp=sharing)
      - label data: [handwritten_mnist_label](https://drive.google.com/file/d/1miFdEi1X8Fr6hZx9_9UbWOmC8MPS27AJ/view?usp=sharing)
  - Fashion MNIST: 
      - training data: [handwritten_mnist_training](https://drive.google.com/file/d/1rEDOowWbCvKFPphJMtoSW0UHvEZYAmxV/view?usp=sharing)
      - label data:  [fashion_mnist_label](https://drive.google.com/file/d/1miFdEi1X8Fr6hZx9_9UbWOmC8MPS27AJ/view?usp=sharing)
#### Pre-trained DNN
- Use pre-trained DNN that has high accuracy. Example:
  - Hand-written digit MNIST: [handwritten_mnist_model](https://drive.google.com/file/d/1eBmWjM3HPp2Ci3e6dhd7iMNYCik2Se8q/view?usp=sharing)
  - Fashion MNIST: [fashion_mnist_model](https://drive.google.com/file/d/1aVk4oMzOSqsh7qzF_zXUC0Qy2ftHmP_B/view?usp=sharing)
### Input Configuration
- Input required configation in file [config.ini](config.ini)
- Example: Please open [config.ini](config.ini)
### Run experiment
#### Run
```sh
python3 main.py
```
#### View Result
- To view summary: open folder `results/hpba/result_summary`
- To access generated advs: open folder `results/hpba/data`
- To access trained autoencode: open folder `resuts/hpba/autoencoder`

**Note**: Result file name contains time stamp
