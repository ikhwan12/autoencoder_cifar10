# Cifar-10 Image Classification using CNN and pre-trained Convolutional Autoencoder

This project report is for fulfilling the second test from Ridge-i company. The topic of this assignment is to classify the image from cifar-10 dataset using supervised and unsupervised learning. In this case, we used autoencoder as a self-learning network to extract some important features from the dataset and using convolutional network to predict the class of the image belong to. VGG style model is the network that I choose as a classifier. The reason to use this model is generally because this model is very deep convolutional network that can be used for large-scale image recognition. More explanation can be found on section 5.

The proposed network is using autoencoder hidden layer that converted into the vectors as extracted features and will be used by classifier for classification. Autoencoder will be trained separately with classifier and lend the encoder layer stacked with the VGG style model. In this experiment, we will compare two different scheme and show that the autoencoder pre-train model can improve the classifier model.

## Getting Started
Install all the dependencies and this experiment works well using python 3.7

```
$ pip install -r requirements.txt
```
## Training
Simply run this script, and if you have pre-trained autoencoder model pass the `-p` paramater and enter your model path. For example, `-p model/autoencoder.h5`
```
$ python main.py --train
```
## Testing
For testing, we will use the cifar-10 dataset that generate automatically by `cifar10.load_data()` which will give us 10,000 data for testing. Please specify your model path by `-p your/model/path` and set your testing data by passing `-lb` for lower bound index and `-ub` for upper bound index of testing dataset. This 2 parameters only accept `int` data type.
```
$ python main.py --test
```
## Evaluation
In order to show some classification evaluation result, we need to run this script with the model path as an input by passing this parameter `-p your/model/path`. This evaluation will show you **test loss**, **test acuracy**, **number of correct and incorrect labels** and **the classification report** that will be saved into `.csv` file
```
$ python main.py --evaluate
```
## Intermediate Activation Layer
You can generate the intermediate activation layer input and output and save it to your directory for further analysis purpose by running this script. As the input you need to passing model path parameter `-p your/model/path`
```
$ python main.py --imshow
```
## Generate History of Training Accuracy and Loss
Simply run this script.
```
$ python result_to_csv.py --json your/json/file --acc <file_name_accuracy> --loss <file_name_loss>
```
## Report
You can see the report on online version by visiting this [link](https://www.ikhwanmiqbal.com/cifar10_classification) or you can download the offline version [here](https://bre.is/JfzduUye)