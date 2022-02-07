# EEG-Classification

### What is this?
In this Project I extracted some features from [BCI2003](https://www.bbci.de/competition/iii/) datas and selected the best features for classifing the signals using PSO (particle swarm optimization). This project was done for EE-SUT computational intelligence course.

### Classification Procedure

First, I maximized the variance between the classes by using CSP filters. Then, I extracted the following features from the trials. Finally, I found the best features whihc have the maximum classification accuracy using PSO and an MLP network.

#### Features
- Mean frequency
- Nedian frequency
- Total power of the channels
- Power of delta, theta, alphas, beta, and gamma frequency bands
- Entropy
- Lyapanov exponent
- Average of differentiaite of trials
- Skewness
- Kurtosis
- Phase of the FFT coefficients

### Finals accuracy
Final accuracy with 5-fold validation was 81%.

### Dependencies

This project uses matlab for feature selection and extraction and uses python for MLP classification. You can install the required libraries for python usin `pip install -r requirements.txt`. Also, first change the path to pyenv in the first lines of main.m .
