# Optical-Character-Recognition-CCPD2019
This repository implements the task of recognizing license plates on the CCPD2019 dataset. A CRNN model was written and trained, in which the output of the convolutional network is fed to the input of the recurrent model. As a result, the quality of 97 percent accuracy and 99 percent CER (Character Error Rate) was obtained on the test data.

model.py - the file in which the model is implemented.

utils.py - the file that implements the functions of calculating quality metrics, encoding and decoding..

main_notebook.ipynb - the file in which the model was launched.
