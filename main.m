clc
clear
load Datas/All_data.mat;
%% Configs
clc
fs = 100;
num_features = 1;
X = x_train;
features = zeros(num_features, size(X, 3), size(X, 2));
%% Feature Extraction
clc
% Mean Freq
n_f = 1;
for  i = 1:size(X, 3)
    features(n_f,i,:) = meanfreq(x_train(:,:,i),fs);
end

% Med Freq
n_f = 2;
for  i = 1:size(X, 3)
    features(n_f,i,:) = medfreq(x_train(:,:,i),fs);
end

% Total Power of Channels
n_f = 3;
for  i = 1:size(X, 3)
    features(n_f,i,:) = bandpower(x_train(:,:,i));
end

% Power of Theta Band
n_f = 4;
freq_band = [4, 8];
for  i = 1:size(X, 3)
    features(n_f,i,:) = bandpower(x_train(:,:,i), fs, freq_band);
end

% Power of alpha Band
n_f = 5;
freq_band = [8, 12];
for  i = 1:size(X, 3)
    features(n_f,i,:) = bandpower(x_train(:,:,i), fs, freq_band);
end

% Power of Beta Band
n_f = 6;
freq_band = [12, 30];
for  i = 1:size(X, 3)
    features(n_f,i,:) = bandpower(x_train(:,:,i), fs, freq_band);
end

% Power of Beta Band
n_f = 7;
freq_band = [30, 50];
for  i = 1:size(X, 3)
    features(n_f,i,:) = bandpower(x_train(:,:,i), fs, freq_band);
end

% Entropy
n_f = 8;
for  i = 1:size(X, 3)
    features(n_f,i,:) = entropy(x_train(:,:,i));
end

% Lyapunov Exponent
n_f = 9;
for  i = 1:size(X, 3)
    features(n_f,i,:) = lyapunovExponent(x_train(:,:,i), fs);
end

% Correlation Dim
n_f = 10;
for  i = 1:size(X, 3)
    features(n_f,i,:) = correlationDimension(x_train(:,:,i));
end














