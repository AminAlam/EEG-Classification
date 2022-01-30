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


% Med Freq
n_f = 2;
for  i = 1:size(X, 3)
    features(n_f,i,:) = medfreq(x_train(:,:,i),fs);
end


