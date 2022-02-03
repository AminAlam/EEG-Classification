%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Phase 1
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
%% extending trials using window with overlap
clc
X_cls_1 = permute(X(:, :, cls1_indexes), [2,1,3]);
X_cls_2 = permute(X(:, :, cls2_indexes), [2,1,3]);
X_cls_1 = reshape(X_cls_1, [28, 7950]);
X_cls_2 = reshape(X_cls_2, [28, 7850]);

n_overlap = 50-25;

X_overlap_cls_1 = zeros(size(X, 1), size(X, 2),  (size(X_cls_1, 2)-50)/n_overlap);
X_overlap_cls_2 = zeros(size(X, 1), size(X, 2),  (size(X_cls_2, 2)-50)/n_overlap);
y_train_overlap = [];
row_counter = 1;
for i = 1:n_overlap:size(X_cls_1, 2)-50
    X_overlap_cls_1(:,:, row_counter) = X_cls_1(:,i:i+49)';
    row_counter = row_counter+1;
    y_train_overlap = [y_train_overlap, 0];
end

row_counter = 1;
for i = 1:n_overlap:size(X_cls_2, 2)-50
    X_overlap_cls_2(:,:, row_counter) = X_cls_2(:,i:i+49)';
    row_counter = row_counter+1;
    y_train_overlap = [y_train_overlap, 1];
end

X_overlap = cat(3, X_overlap_cls_1, X_overlap_cls_2);

% shuffling classes

indexes = randperm(size(X_overlap, 3));
X_overlap = X_overlap(:, :, indexes);
y_train_overlap = y_train_overlap(:, indexes);

X = X_overlap;
y_train = y_train_overlap;
%% Feature Extraction
clc
[features, num_features] = feature_extraction(X, fs);
%% Calculating the 2D Fisher Score
clc
selected_features = fisher_2D_calculator(num_features, features, cls1_indexes, cls2_indexes);
%% Calculating the ND Fisher Score
clc
k = 16;
num_itters = 400;
best_features = fisher_ND_calculator(k, num_itters, selected_features, cls1_indexes, cls2_indexes);
%% MLP 
clc
close all
K_folds = 5;
lr = 0.001;
num_epochs = 400;

[loss_train, loss_val, acc_train, acc_val] = MLP(best_features, K_folds, lr, num_epochs, y_train);

plot(loss_train, 'LineWidth', 2)
hold on
plot(loss_val, 'LineWidth', 2)
legend('train', 'val')
title('Loss')

figure
plot(acc_train, 'LineWidth', 2)
hold on
plot(acc_val, 'LineWidth', 2)
ylim([0,100])
legend('train', 'val')
title('Accuracy')

disp(sprintf("Training Acc %f ",acc_train(end)))
disp(sprintf("Validation Acc %f ",acc_val(end)))
%% RFB
clc
close all

[acc_train, acc_val] = RBF(K_folds, best_features, y_train);

disp(sprintf("Training Acc %f ",mean(acc_train)))
disp(sprintf("Validation Acc %f ",mean(acc_val)))
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Phase 2 
% PSO
clc
close all

num_epochs = 400;
num_fishes = 20;
time = 20;
alpha = 0.2;

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
    tic
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
        
        alpha = alpha - 0.001;
        B1 = rand()*alpha;
        B2 = rand()*alpha;
        
        v(f, :) = alpha*v(f, :)+B1*(x_local(f,:)-x(f,:))+B2*(x_global-x(f,:));
        x(f, :) = round(x(f, :) + v(f, :))';
        
        % correct bad x values which cross the limits
        [~, col] = find(x(f, :)>size(selected_features, 1));
        if ~isempty(col)
            x(f, col) = size(selected_features, 1);
        end

        [~, col] = find(x(f, :)<1);
        
        if ~isempty(col)
            x(f, col) = 1;
        end
    end
    q
    toc
end

%% Genetric Algorithm
clc
close all





























