# EEG-Classification

### What is this?
In this Project, I extracted some features from [BCI2003](https://www.bbci.de/competition/iii/) data and selected the best features for classifying the signals using PSO (particle swarm optimization). This project was done for the EE-SUT computational intelligence course.

### Classification Procedure

First, I maximized the variance between the classes using CSP filters. Then, I extracted the following features from the trials. Finally, I found the best features which have the maximum classification accuracy using PSO and an MLP network.

#### Features
- Mean frequency
- Median frequency
- Total power of the channels
- Power of delta, theta, alphas, beta, and gamma frequency bands
- Entropy
- Lyapunov exponent
- Average of differentiate of trials
- Skewness
- Kurtosis
- Phase of the FFT coefficients

### Finals accuracy
Final accuracy with 5-fold validation was 87%.

### Dependencies

This project uses Matlab for feature selection and extraction and uses python for MLP classification. You can install the required libraries for python using `pip install -r requirements.txt`. Also, first, change the path to pyenv in the first lines of main.m.
