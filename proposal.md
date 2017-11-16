# Machine Learning Engineer Nanodegree
## Capstone Proposal
Kaiyuan Hu
October 11, 2017

## Kaggle competition: Statoil/C-CORE Iceberg Classifier Challenge

### Domain Background

[Statoil/C-CORE Iceberg Classifier Challenge](https://www.kaggle.com/c/statoil-iceberg-classifier-challenge) is an ongoing competition state on kaggle.

_Statoil_ worked closely with _C-CORE_ that use remote sensing system to detect icebergs. The remote sensing system are housed on satellites over 600 kilometers above the earth. The satellites provide two channels image information: HH(transmit/receive hotizontally) and HV(transmit horizontally and receive vertically). They want to use those image data to find a model can classify those icebergs from ships.

Image classification is a difficult problem in computer vision area. Convolutional neural network(CNN) is a successful method in deal with image classification issues. In 2015, K.He, X.Zhang, etc proposed a famous CNN architecture named [ResNet-152](https://arxiv.org/abs/1512.03385) that achieve 95.51% accuracy in image classification on [ImageNet](http://www.image-net.org/) which is a superhuman result. In this icebergs and ships image classification problem. I want to propose a CNN architecture to solve this problem.

### Problem Statement

Since this is an ongoing competition state on kaggle, this problem does not been solved. This is a binary image classification problem, the best result up-to-date on kaggle have 0.1029 log score and the 20th best result on kaggle have 0.1463 log score. So the porfermance of model can be evaluated by log score.

### Datasets and Inputs

_Statoil_ and _C-CORE_ provide a training dataset with size 42.85mb and a test dataset with size 245.22mb. The format of the data is in json. For each row, the data have 5 different name fields.

- id: the id of the Image
- band_1, band_2: the flattened image data. Each band has 75*75 pixel values in the list, so in each band there is a list with 75\*75=5625 elements. Each element is a float number.
- inc_angle: the incidence angle of which the image was take. The field have missing data marked 'na'.
- is_iceberg: this is the target variable, set to 1 if there is an iceberg and set to 0 if it is a ship.

Based on the information above, we know that the image is black-white scale and I will mainly use band_1 and band_2 those two variables.

Here is the sample image.

![icebergs](/Users/Kieran/Documents/MLND/Statoil-C-CORE-Iceberg-Classifier-Challenge/sample_img/8ZkRcp4.png)
![ship](/Users/Kieran/Documents/MLND/Statoil-C-CORE-Iceberg-Classifier-Challenge/sample_img/nXK6Vdl.png)

### Solution Statement

Since this is a binary image classification problem, CNN seems suitable for this poblem. I will gonne to design a transfer learning model that use two pre_trained ResNet50 model first. After that mix the result from the pre_trained model and add some other layers like dropout layer maxpooling layer and fully connected layer. Based on my knowledge, I don' t think the incidence angle will matter but I will try to add this feature if I have enough time.

Here is the detail:
- I will use two pre_trained Resnet50 models first. One is to handle band_1 and another is to handle band_2.
- I may add some pooling layers and dropout layers after each pre_trained Resnet50 model,
- I will use a Concatenate layer to merge two model.
- After the Concatenate layer, I may add few more layers, like fully connected layer, dropout layer and pooling layer.
- Then I will add a fully connected layer with sigmoid activation funtion.

Basically I will follow the idea state above, but this is just a statement. I may modify the structure of the model during the implementation.

### Benchmark Model

[noobhound](https://www.kaggle.com/knowledgegrappler/a-keras-prototype-0-21174-on-pl) proposed a CNN model on Kaggle. This CNN model reach 0.21174 log score. The model _noobhound_ used seems to be a simplified VGG-16 net. He add some dropout layer between each block and he change the final fully connected layer to GlobalMaxPooling layer. He also add a Concatenate step that mix band_1 and band_2. You can find the detail implementation of he model at [Here](https://www.kaggle.com/knowledgegrappler/a-keras-prototype-0-21174-on-pl). The model is in code block 4.

### Evaluation Metrics

The porfermance of this model can be evaluated by accuracy score.

Here is a sample table:


|                     | Iceberg     | Ship |
| ------------------- |-------------| -----|
| Prediction accuracy |             |      |


### Project Design
_(approx. 1 page)_

1. Clean the data.
2. visualize the image.
3. Design the CNN models.
4.

In this final section, summarize a theoretical workflow for approaching a solution given the problem. Provide thorough discussion for what strategies you may consider employing, what analysis of the data might be required before being used, or which algorithms will be considered for your implementation. The workflow and discussion that you provide should align with the qualities of the previous sections. Additionally, you are encouraged to include small visualizations, pseudocode, or diagrams to aid in describing the project design, but it is not required. The discussion should clearly outline your intended workflow of the capstone project.

### Reference
- [He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)
- https://www.kaggle.com/knowledgegrappler/a-keras-prototype-0-21174-on-pl
- https://www.kaggle.com/c/statoil-iceberg-classifier-challenge
