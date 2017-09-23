# Traffic-Sign-Classifier
A Traffic Sign Classifier trained on the German Traffic Sign Dataset (43 classes). Network based on LeNet.

## Dependencies

* pickle
* numpy
* matplotlib
* sklearn
* tensorflow
* glob
* scipy
* pillow

run 
`python gtsc.py `
for training to testing full length code

run 
`python imageTest.py`
to run the trained model for classification

my_data folder contains test images downloaded from web in jpg format, dont change folder name,
file name does not matter as long as .jpg.
data folder contains pickled data set of train, test and validation data provided by german traffic
sign data set.
lenet.* files are saved copies of tensor flow training session.
signnames.csv has an annotation of labels/classes in german traffic sign dataset.
