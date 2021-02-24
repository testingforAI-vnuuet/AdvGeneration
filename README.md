# AdvGeneration

Contain source code of adversarial example generation methods. We try to re-implement existing adversarial example
generation methods, then compare these methods with our proposed method. For simplicity, the proposed method is named
AE4DNN (Autoencoder For attack Deep Neural Network).

If you have any questions, please contact me via nguyenducanh@vnu.edu.vn.

## 1. Installation

Before running, it is mandatory to install necessary packages, which is defined in requirements.txt. Depending on your
preference, there are two main ways:

### 1.1. PyCharm

Refer to <a href="https://www.jetbrains.com/help/pycharm/managing-dependencies.html"> this link</a> for further
information.

### 1.2. Terminal (such as HPC system)

- Step 1. Change directory to the folder containing requirements.txt

- Step 2: Type: **pip install -r requirements.txt**. This command might take for a few minutes to accomplish.

## 2. Run

Assume you want to run autoencoder.py

### 2.1. PyCharm

Just run directly on GUI

### 2.2. Terminal (such as HPC system)

- Step 1. Set up PYTHONPATH to update search path for module files. Without this command, python will struggle with
  finding import modules. For example, execute

**export PYTHONPATH=/home/anhnd/AdvGeneration/src:$PYTHONPATH**

to add **home/anhnd/AdvGeneration/src** to search path.

- Step 2. Change directory to the folder containing the executing file
- Step 3. Type: **python autoencoder.py**

## 3. Additional experiment.

Please read this document ([additional_experiment.pdf](https://drive.google.com/file/d/1PUN_Rel_VyAlJhMkdwdPH-CUL8paMBzM/view?usp=sharing)) for additional experiment.

[comment]: <> (## 3. Experimental results)

[comment]: <> (AE4DNN is compared with FGSM, l.l. class, box-constrained L-BFGS,)

[comment]: <> (Carnili-Wagner <img src="https://render.githubusercontent.com/render/math?math=||L||_2"> attack, and AAE to demonstrate)

[comment]: <> (how it mitigates the trade-off and unstable transferable rate. Specifically, the experiment addresses the following)

[comment]: <> (research questions:)

[comment]: <> (- Does AE4DNN produce high quality of adversaries compared to other methods? &#40;RQ1&#41;)

[comment]: <> (- Does AE4DNN require low computational cost compared to other methods? &#40;RQ2&#41;)

[comment]: <> (- Does AE4DNN effectively when dealing with a set of new input vectors? &#40;generalization ability&#41; &#40;RQ3&#41;)

[comment]: <> (- Does the generated adversaries from AE4DNN benefit for attacking other models? &#40;transferable ability&#41; &#40;RQ4&#41;)

[comment]: <> (The research chooses MNIST which is a popular publicly-available dataset for evaluation. The training set contains)

[comment]: <> (50,000 samples. The test set has 10,000 samples. Each sample on the dataset is an image with 28 pixels in width and 28)

[comment]: <> (pixels in height. The value of each pixel is in range of 0 and 255, which indicates the lightness or darkness of that)

[comment]: <> (pixel. Adversarial example in this experiment is called adversarial image for simplicity.)

[comment]: <> (### 3.1. Quality of adversaries in terms of <img src="https://render.githubusercontent.com/render/math?math=||L||_2"> distance.)

[comment]: <> (In practice, machine learning testers have no idea about the best value of configurations. Therefore, the testers)

[comment]: <> (usually use the strategy try-and-check until they find out the optimal configuration. The experiment in this section)

[comment]: <> (follows this strategy. For FGSM, the value of 𝜖 changed from 0.1 to 0.3 with a step of 0.05. For least likely class,)

[comment]: <> (the value of 𝜖 changed from 0.1 to 0.3 with a step of 0.03 and the number of iterations is 4. Concerning)

[comment]: <> (box-constrained L-BFGS, the value of 𝜖 changed from 0.001 to 0.0035 with a step of 0.0005 and the number of iterations)

[comment]: <> (is 20.)

[comment]: <> (![box plots]&#40;./images/box_plotsv3.png&#41;)

[comment]: <> (### 3.2 Experiment with other autoencoder architectures)

[comment]: <> (#### 3.2.1 Autoencoder 1)

[comment]: <> (The architecture is described as follow:)

[comment]: <> (The average||L||2 distance and the corresponding number of adversaries with different values of β in AE4DNN. Good values)

[comment]: <> (of β are marked in bold)

[comment]: <> (|     β   | 0.0005 | 0.001 | 0.002 | 0.003 | 0.004 | 0.005 |)

[comment]: <> (|:------:|:------:|:-----:|:-----:|:-----:|:-----:|:-----:|)

[comment]: <> (| AVG L2 |   6.4  |  6.4  |  6.62 |  6.71 |  6.6  |  6.73 |)

[comment]: <> (|  # adv |   825  |  856  |  876  |  883  |  883  |  882  |)

[comment]: <> (The average||L||2 distance and the corresponding number of adversaries with different values of φ &#40;AAE&#41;.)

[comment]: <> (|     φ   | 0.01 | 0.02 | 0.03 | 0.04 | 0.05 |)

[comment]: <> (|:------:|:----:|:----:|:----:|:----:|:----:|)

[comment]: <> (| AVG L2 | 6.11 | 6.32 | 6.47 | 6.49 | 6.65 |)

[comment]: <> (|  # adv |  99  |  652 |  827 |  843 |  879 |)

[comment]: <> (The comparison between AE4DNN and AAE in terms of generalization. Target label is 7. Better values are marked in bold. The)

[comment]: <> (total time to perform 10k-attack, 20k-attack, and 40k-attack are approximate to 1.6 seconds, 3.1 seconds, and 6.3)

[comment]: <> (second, respectively.These attacks do not need to train the autoencoder.)

[comment]: <> (| Config |      &#40;β, φ&#41;      |)

[comment]: <> (|:------:|:--------------:|)

[comment]: <> (|    A   | &#40;0.0005, 0.03&#41; |)

[comment]: <> (|    B   |  &#40;0.0005, 0.04 |)

[comment]: <> (|    C   |  &#40;0.002, 0.05&#41; |)

[comment]: <> (| Config |    Average L2    |  | Average Adversarial rate &#40;%&#41; |      |)

[comment]: <> (|:------:|--------|:----------:|------------------------------|------|)

[comment]: <> (|        | AE4DNN | AAE        | AE4DNN                       | AAE  |)

[comment]: <> (|    A   | **6.49**   |    6.57    | 82.8                         | **82.9** |)

[comment]: <> (|    B   | **6.49**   |     6.6    | 82.8                         | **84.4** |)

[comment]: <> (|    C   | **6.7**    |    6.76    | 88.7                         | **89.1** |)

[comment]: <> (The transferable rate between AE4DNN and AAE with different DNN models. Better values are marked in bold)

[comment]: <> (| Config | VGG-13 &#40;%&#41; |      | VGG-16 &#40;%&#41; |      | LeNet-5 &#40;%&#41; |      | AlextNet &#40;%&#41; |     |)

[comment]: <> (|:------:|------------|:----:|------------|------|-------------|------|--------------|-----|)

[comment]: <> (|        | AE4DNN     | AAE  | AE4DNN     | AAE  | AE4DNN      | AAE  | AE4DNN       | AAE |)

[comment]: <> (|    A   | 18         | 13.8 | 21.2       | 11   | 11.8        | 0.8  | 0.2          | 2.8 |)

[comment]: <> (|    B   | 18         | 36.4 | 21.2       | 43.5 | 11.8        | 11.7 | 0.2          | 2.6 |)

[comment]: <> (|    C   | 27         | 51.6 | 30.4       | 36.4 | 2.9         | 4.9  | 3.9          | 0.9 |)

[comment]: <> (#### 3.2.2 Autoencoder 2)

[comment]: <> (The architecture is described as follow:)

[comment]: <> (The average||L||2 distance and the corresponding number of adversaries with different values of β in AE4DNN. Good values)

[comment]: <> (of β are marked in bold)

[comment]: <> (|     β   | 0.0005 | 0.001 | 0.002 | 0.003 | 0.004 | 0.005 |)

[comment]: <> (|:------:|:------:|:-----:|:-----:|:-----:|:-----:|:-----:|)

[comment]: <> (| AVG L2 |  4.87  |  6.82 |  6.77 |  7.74 |  7.57 |  7.49 |)

[comment]: <> (|  # adv |   553  |  870  |  872  |  883  |  883  |  880  |)

[comment]: <> (The average||L||2 distance and the corresponding number of adversaries with different values of φ &#40;AAE&#41;.)

[comment]: <> (|    φ    | 0.01 | 0.02 | 0.03 | 0.04 | 0.05 |)

[comment]: <> (|:------:|:----:|:----:|:----:|:----:|:----:|)

[comment]: <> (| AVG L2 | 4.25 | 4.92 | 6.16 | 7.55 | 8.12 |)

[comment]: <> (|  # adv |  39  |  159 |  607 |  782 |  883 |)

[comment]: <> (The comparison between AE4DNN and AAE in terms of generalization. Target label is 7. Better values are marked in bold. The)

[comment]: <> (total time to perform 10k-attack, 20k-attack, and 40k-attack are approximate to 1.6 seconds, 3.1 seconds, and 6.3)

[comment]: <> (second, respectively.These attacks do not need to train the autoencoder.)

[comment]: <> (| Config |      &#40;β, φ&#41;      |)

[comment]: <> (|:------:|:--------------:|)

[comment]: <> (|    A   | &#40;0.0005, 0.02&#41; |)

[comment]: <> (|    B   |  &#40;0.003, 0.04 |)

[comment]: <> (|    C   |  &#40;0.004, 0.04&#41; |)


[comment]: <> (| Config | Average L2|            |   Average Adversarial rate &#40;%&#41;     |  |)

[comment]: <> (|:------:|:------:|:----------:|:------:|:----------------------------:|)

[comment]: <> (|        | AE4DNN |     AAE    | AE4DNN |              AAE             |)

[comment]: <> (|    A   |  **4.99**  |    5.14    |  **55.35** |             16.37            |)

[comment]: <> (|    B   |  7.85  |    **7.62**    |  **89.47** |             78.32            |)

[comment]: <> (|    C   |  7.67  |    **7.62**    |  **89.09** |             78.32            |)

[comment]: <> (The transferable rate between AE4DNN and AAE with different DNN models. Better values are marked in bold)

[comment]: <> (| Config | VGG-13 &#40;%&#41; |      | VGG-16 &#40;%&#41; |      | LeNet-5 &#40;%&#41; |      | AlextNet &#40;%&#41; |     |)

[comment]: <> (|:------:|------------|:----:|------------|------|-------------|------|--------------|-----|)

[comment]: <> (|        | AE4DNN     | AAE  | AE4DNN     | AAE  | AE4DNN      | AAE  | AE4DNN       | AAE |)

[comment]: <> (|    A   | **18.5**   | 12.3 | **43.8**   | 16   | **4.3**        | 3.3  | **18.1**          | 12.2 |)

[comment]: <> (|    B   | **67.3**   | 44.1 | **77.5**   | 54.1 | **10.8**        | 5.7 | **59.5**          | 44.6 |)

[comment]: <> (|    C   | 40.8       | **44.1** | **81.8**| 54.1 | 4.5         | **5.7**  | **55.25**          | 44.6 |)

## 4. Examples

The figure below shows some adversaries generated by the proposed method and comparable methods.

![image results](./images/best_adv.png)

[comment]: <> (# Useful command)

[comment]: <> (### download file 0_to_1.png from server)

[comment]: <> (scp -P 22033 anhnd@uet-hpc.remote.hpc.farm:/home/anhnd/AdvGeneration/data/mnist/model/0_to_1.png)

[comment]: <> (/Users/ducanhnguyen/Documents)

[comment]: <> (### download folder model from server)

[comment]: <> (scp -P 22033 -r anhnd@uet-hpc.remote.hpc.farm:/home/anhnd/AdvGeneration/data/mnist/model /Users/ducanhnguyen/Documents)

[comment]: <> (### upload local folder src to server)

[comment]: <> (scp -P 22033 -r /Users/ducanhnguyen/Documents/PycharmProjects/AdvGeneration/src anhnd@uet-hpc.remote.hpc.farm:)

[comment]: <> (/home/anhnd/AdvGeneration)

