clc
clear
clear class
load Datas/All_data.mat;
%% Configs
clc
fs = 100;
X = x_train;
[~, cls1_indexes] = find(y_train==0);
[~, cls2_indexes] = find(y_train==1);
%% Feature Extraction
clc
[features, num_features] = feature_extraction(X, fs);
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
k = 15;
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

[~, col] = find(FS_ND == max(FS_ND));
best_features_indexes = selected_samples_memory(col(1), :);
best_features = selected_features(best_features_indexes,:);
%% changin the python enviornment - Matlab 2019 works with python 3.7
datas
env_path = '/Users/mohammadaminalamalhod/anaconda3/envs/Matlab/bin/python3.7';
pyenv('Version',env_path)
%% MLP 
clc
close all
K_folds = 6;

num_datas_in_fold = floor(size(best_features,2)/K_folds);

for f = 0:K_folds-1
    f
    if f ==0
        folds_train_1 = [];
        folds_val = best_features(:,f*num_datas_in_fold+1:(f+1)*num_datas_in_fold);
        folds_train_2 = best_features(:,(f+1)*num_datas_in_fold+1:end);
    elseif f == K_folds-1
        folds_train_1 = best_features(:,1:f*num_datas_in_fold);
        folds_val = best_features(:,f*num_datas_in_fold+1:(f+1)*num_datas_in_fold);
        folds_train_2 = [];
    else
        folds_train_1 = best_features(:,1:f*num_datas_in_fold);
        folds_val = best_features(:,f*num_datas_in_fold+1:(f+1)*num_datas_in_fold);
        folds_train_2 = best_features(:,(f+1)*num_datas_in_fold+1:end);
    end
    
%     
%     datas_train = py.numpy.array(best_features(:,1:300)');
%     datas_val = py.numpy.array(best_features(:,301:end)');
%     labels_train = py.numpy.array(y_train(:,1:300)');
%     labels_val = py.numpy.array(y_train(:,301:end)');
% 
%     kwa = pyargs('datas_train', datas_train, ...
%         'datas_val', datas_val, ...
%         'labels_train', labels_train, ...
%         'labels_val', labels_val, ...
%         'input_size', int32(k), ...
%         'ouput_size', int32(1));
% 
%     mod = py.importlib.import_module('classifier');
%     py.importlib.reload(mod);
%     out = mod.call_from_matlab(kwa);
% 
%     loss_train = str2num(char(out.cell{1}));
%     loss_val = str2num(char(out.cell{2}));
%     acc_train = str2num(char(out.cell{3}));
%     acc_val = str2num(char(out.cell{4}));

end

plot(loss_train)
hold on
plot(loss_val)
legend('train', 'val')
figure
plot(acc_train)
hold on
plot(acc_val)
ylim([0,100])
legend('train', 'val')








