# EQ2341 Final project
### Hang Qin, Chenting Zhang

## Project structure
The project contains three modules and one main script. Each module is represented as a folder.

## main.py

The hmm training function (EM) is implemented in main.py

Running the main.py will execute specific times of training iteration, then perform testing on test data 
using the hmm based on the trained parameters.

## data
All data can be found under the data folder.

The two .txt files are the original data.

And the two .npy files are the loaded data that have been already transformed into matrix form.

## PattRecClasses

The python classes that are necessary for building a HMM model.

## utils

This folder contains two scripts. 

load_data.py read the .txt data and transform it into numpy array and save them.

functionality.py includes two functions. 

The likelihood function can compute b_j(x_t), which are necessary for HMM training and testing. 
The hmm_test function is the test function using hmm on the given observed data.


