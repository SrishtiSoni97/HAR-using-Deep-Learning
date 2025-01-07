# HAR-using-Deep-Learning
Human Activity Recognition using DL models like CNN, RNN, &amp; DNN
## Introduction
This repository is to apply deep learning models on Human Activity Recognition(HAR)/Activities of Daily Living(ADL) datasets. Three deep learning models, including Convolutional Neural Networks(CNN), Deep Feed Forward Neural Networks(DNN) and Recurrent Neural Networks(RNN) were applied to the datasets. Six HAR/ADL benchmark datasets were tested. The goal is to gain some experiences on handling the sensor data as well as classifying human activities using deep learning. 

## Benchmark datasets
  * [UCI HAR dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) contains data of 6 different physical activities walking, walking upstairs, walking downstairs, sitting, standing and laying), performed by 30 subjects wearing a smartphone (Samsung Galaxy S II) on the waist.

## Apporach
  * For each dataset, a slicing window appoarch was applied to segment the dataset. Each segment includes a series of data (usually 25 sequential data points) and two continuous windows have 50% overlapping. 
  * After data preprocessing which includes reading files, data cleaning, data visualization, relabling and data segmentation, the data was saved into hdf5 files.
  * Deep learning models including CNN, DNN and RNN were applied. For each model in each dataset, hyperparameters were optimized to get the best performance.
  * To combine the data from multimodalities, different data fusion methods were applied on Sphere and SHL dataset.
## Dependencies
* Python 3.10

## Usage
No need to download the UCI dataset, it's already present in the repository.

## Note 
I am working toward using Federated Learning approach to implement HAR.

