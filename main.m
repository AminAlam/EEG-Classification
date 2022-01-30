clc
clear
load Datas/All_data.mat;
%% Configs
clc
fs = 100;
num_features = 10;
X = x_train;
%% Feature Extraction
clc
% Mean Freq
features = feature_extraction(X, num_features, fs);