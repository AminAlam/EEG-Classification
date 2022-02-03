%% Phase 1
clc
clear
clear class
load Datas/All_data.mat;
%% changin the python enviornment - Matlab 2019 works with python 3.7
env_path = '/Users/mohammadaminalamalhod/anaconda3/envs/Matlab/bin/python3.7';
pyenv('Version',env_path);
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
selected_features = fisher_2D_calculator(num_features, features, cls1_indexes, cls2_indexes);
%% Calculating the ND Fisher Score
clc
k = 15;
num_itters = 200;
best_features = fisher_ND_calculator(k, num_itters, selected_features, cls1_indexes, cls2_indexes);
%% MLP 
clc
close all
K_folds = 5;
lr = 0.001;
num_epochs = 400;

[loss_train, loss_val, acc_train, acc_val] = MLP(best_features, K_folds, lr, num_epochs, y_train);

plot(mean_loss_train, 'LineWidth', 2)
hold on
plot(mean_loss_val, 'LineWidth', 2)
legend('train', 'val')
title('Loss')

figure
plot(mean_acc_train, 'LineWidth', 2)
hold on
plot(mean_acc_val, 'LineWidth', 2)
ylim([0,100])
legend('train', 'val')
title('Accuracy')

disp(sprintf("Training Acc %f ",mean_acc_train(end)))
disp(sprintf("Validation Acc %f ",mean_acc_val(end)))
%% RFB
clc
close all

acc_train = [];
acc_val = [];


for f = 0:K_folds-1
    
    if f == 0
        folds_train_1 = [];
        folds_val = f*num_datas_in_fold+1:(f+1)*num_datas_in_fold;
        folds_train_2 = (f+1)*num_datas_in_fold+1:num_all_datas;
    elseif f == K_folds-1
        folds_train_1 = 1:f*num_datas_in_fold;
        folds_val = f*num_datas_in_fold+1:(f+1)*num_datas_in_fold;
        folds_train_2 = [];
    else
        folds_train_1 = 1:f*num_datas_in_fold;
        folds_val = f*num_datas_in_fold+1:(f+1)*num_datas_in_fold;
        folds_train_2 = (f+1)*num_datas_in_fold+1:num_all_datas;
    end
    folds_train = [folds_train_1, folds_train_2];
    
    
    datas_train = best_features(:,folds_train);
    datas_val = best_features(:,folds_val);
    labels_train = y_train(:,folds_train);
    labels_val = y_train(:,folds_val);
    
    net = newrbe(datas_train,labels_train, 'spread', 4);
    val_out = net(datas_val);
    train_out = net(datas_train);
    [row, col] = find(floor(train_out)==labels_train);
    acc_train = [acc_train, length(row)/length(train_out)];
    [row, col] = find(floor(val_out)==labels_val);
    acc_val = [acc_val, length(row)/length(val_out)];
    

end

disp(sprintf("Training Acc %f ",mean(acc_train)))
disp(sprintf("Validation Acc %f ",mean(acc_val)))
%% Phase 2 
% PSO
clc
close all

num_fishes = 100;
time = 100;

v = zeros(num_fishes, time, k);
x = zeros(num_fishes, time, k);

x_local = zeros(num_fishes, k);
x_global = zeros(1,k);

alpha = 0.1;

for t = 1:time
    for f = 1:num_fishes
        B1 = rand();
        B2 = rand();
        
        
        
    end
    alpha = alpha - 0.001;
end











