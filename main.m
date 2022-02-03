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

[acc_train, acc_val] = RBF(K_folds, best_features, y_train);

disp(sprintf("Training Acc %f ",mean(acc_train)))
disp(sprintf("Validation Acc %f ",mean(acc_val)))
%% Phase 2 
% PSO
clc
close all

num_fishes = 10;
time = 20;
alpha = 0.1;

k = 16;
v = zeros(num_fishes, k);
x = zeros(num_fishes, k);

x_local = zeros(num_fishes, k);
x_global = zeros(1,k);
q = -inf;

for f = 1:num_fishes
    x(f,:) = randsample(1:size(selected_features, 1), k);
    x_local(f,:) = x(f,:);
    
    [~, ~, ~, acc_val_x] = MLP(selected_features(x(f,:), :), K_folds, lr, num_epochs, y_train);
    acc_val_x = acc_val_x(end);
    
    if acc_val_x > q
        x_global = x(f,:);
        q = acc_val_x;
    end
    
end


for t = 1:time
    for f = 1:num_fishes
        [~, ~, ~, acc_val_x] = MLP(selected_features(x(f,:), :), K_folds, lr, num_epochs, y_train);
        acc_val_x = acc_val_x(end);
        [~, ~, ~, acc_val_x_local] = MLP(selected_features(x_local(f,:), :), K_folds, lr, num_epochs, y_train);
        acc_val_x_local = acc_val_x_local(end);
        if acc_val_x >= acc_val_x_local
            x_local(f,:) = x(f,:);
        end
        if acc_val_x >= q
            q = acc_val_x;
            x_global = x(f,:);
        end
    end
    for f = 1:num_fishes
        B1 = rand();
        B2 = rand();
        alpha = alpha - 0.0002;
        v(f, :) = alpha*v(f, :)+B1*(x_local(f,:)-x(f,:))+B2*(x_global-x(f,:));
        x(f, :) = round(x(f, :) + v(f, :));
    end
    
    q
end
%%


for t = 1:time-1
    for f = 1:num_fishes
        v(f, t+1, :) = alpha*squeeze(v(f, t, :))'+B1*(x_local(f,:)-squeeze(x(f, t, :))')+B2*(x_global-squeeze(x(f, t, :))');
        x(f, t+1, :) = x(f, t, :)+v(f, t, :);
        
        
        
    end
    alpha = alpha - 0.001;
end











