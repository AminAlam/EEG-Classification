clc
clear
load Datas/All_data.mat;
%% Configs
clc
fs = 100;
num_features = 13;
X = x_train;
[~, cls1_indexes] = find(y_train==0);
[~, cls2_indexes] = find(y_train==1);
%% Feature Extraction
clc
features = feature_extraction(X, num_features, fs);
%% Calculating the 2D Fisher Score
clc
FS = zeros(num_features, size(features,3));
for ch = 1:size(features,3)
    for f = 1:num_features
       sigg_cls1 = features(f, cls1_indexes, ch);
       sigg_cls2 = features(f, cls2_indexes, ch);
       FS(f, ch) = fisher_score_2D(sigg_cls1, sigg_cls2, features(f, :, ch));
    end
end

mean_fisher = mean(FS, 'all');
var_fisher = var(FS, 0 , 'all');
thresh = 2*mean_fisher;
thresh = 0.01;
[row, col] = find(FS>thresh);
selected_features = zeros(length(row), size(features,2));
for i = 1:length(row)
    selected_features(i,:) = features(row(i), :, col(i));
end
selected_features = cell2mat(arrayfun(@(x) features(row(x), :, col(x)), 1:length(row),'UniformOutput',false)');

%% Calculating the ND Fisher Score
clc
k = 12;
num_itters = 100;
selected_samples_memory = zeros(num_itters, k);
FS_ND = zeros(1,100);
for itter = 1:num_itters
   samples_feature_indexes = randsample(1:size(selected_features, 1), k);
   selected_samples_memory(itter, :) = samples_feature_indexes;
   sigg_cls1 = selected_features(samples_feature_indexes, cls1_indexes);
   sigg_cls2 = selected_features(samples_feature_indexes, cls2_indexes);
   FS_ND(1, itter) = fisher_score_ND(sigg_cls1, sigg_cls2, selected_features(samples_feature_indexes, :));
end

max(FS_ND)


%%
samples_feature_indexes = [18,12,16,2,35,5,28,7,20,14,25,11];
selected_samples_memory(itter, :) = samples_feature_indexes;
sigg_cls1 = selected_features(samples_feature_indexes, cls1_indexes);
sigg_cls2 = selected_features(samples_feature_indexes, cls2_indexes);
fisher_score_ND(sigg_cls1, sigg_cls2, selected_features(samples_feature_indexes, :))
