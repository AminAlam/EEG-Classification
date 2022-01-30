clc
clear
load Datas/All_data.mat;
%% Configs
clc
fs = 100;
num_features = 10;
X = x_train;
[~, cls0_indexes] = find(y_train==0);
[~, cls1_indexes] = find(y_train==1);
%% Feature Extraction
clc
features = feature_extraction(X, num_features, fs);
%% Calculating the Fisher Score
clc
FS = zeros(no_feature, size(features,3));
for ch = 1:size(features,3)
    for f = 1:no_feature
       sigg_cls0 = features(f, cls0_indexes, ch);
       sigg_cls1 = features(f, cls1_indexes, ch);
       FS(f, ch) = fisher_score(sigg_cls0, sigg_cls1, features(f, :, ch));
    end
end