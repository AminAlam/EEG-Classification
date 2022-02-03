%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Phase 1
clc
clear
clear class
close all
load Datas/All_data.mat;

save_figures = 1;
%% changin the python enviornment - Matlab 2019 works with python 3.7
env_path = '/Users/mohammadaminalamalhod/anaconda3/envs/Matlab/bin/python3.7';
pyenv('Version',env_path);
%% Configs
clc
fs = 100;
X = x_train;
[~, cls1_indexes] = find(y_train==0);
[~, cls2_indexes] = find(y_train==1);
%% Plot avg of all trials and all channels in time and freq domain
clc
close all

plot_time_freq(X, cls1_indexes, cls2_indexes, fs)
% if save_figures
%     set(gcf,'PaperPositionMode','auto')
%     print("Report/images/amp",'-dpng','-r0')
% end

% if save_figures
%     set(gcf,'PaperPositionMode','auto')
%     print("Report/images/stft",'-dpng','-r0')
% end
%% CSP
clc
close all

X = x_train;
C_x1 = 0;
for i = cls1_indexes
    C_x1 = C_x1+ X(:,:,i)'*X(:,:,i);
end

C_x2 = 0;
for i = cls2_indexes
    C_x2 = C_x2+ X(:,:,i)'*X(:,:,i);
end


[V,D] = eig(C_x1, C_x2);
D = diag(D);
[D, indx] = sort(D, 'descend');
V = (V(:,indx));
W1 = V(:,1);
Wend = V(:,end);
X_filt = X;

for i = cls1_indexes
    X_filt(:,:,i) = (V'*X(:,:,i)')';
end

for i = cls2_indexes
    X_filt(:,:,i) = (V'*X(:,:,i)')';
end

plot_time_freq(X_filt, cls1_indexes, cls2_indexes, fs)
X = X_filt;
%% Feature Extraction
clc
close all
[features, num_features] = feature_extraction(X, fs);
%% plot features
f_index = 12;
scatter(mean(features(f_index,cls1_indexes,:), 3), cls1_indexes*0, 'r')
hold on
scatter(mean(features(f_index,cls2_indexes,:), 3), cls2_indexes*0+1, 'b')
%% Calculating the 2D Fisher Score
clc
selected_features = fisher_2D_calculator(num_features, features, cls1_indexes, cls2_indexes);
%% Calculating the ND Fisher Score
clc
k = 16;
num_itters = 800;
best_features = fisher_ND_calculator(k, num_itters, selected_features, cls1_indexes, cls2_indexes);
%% MLP
clc
close all
K_folds = 5;
lr = 0.005;
num_epochs = 50;

[loss_train, loss_val, acc_train, acc_val] = MLP(best_features, K_folds, lr, num_epochs, y_train);


subplot(2,1,1)
plot(loss_train, 'LineWidth', 2)
hold on
plot(loss_val, 'LineWidth', 2)
legend('train', 'val')
title('Loss')

subplot(2,1,2)
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

lr = 0.005;
num_epochs = 50;
num_fishes = 10;
time = 40;
a = 1;
c = 0.4;

k = size(selected_features, 1); %16;
v = zeros(num_fishes, k);
x = zeros(num_fishes, k);

x_local = zeros(num_fishes, k);
x_global = zeros(1,k);
q = -inf;

for f = 1:num_fishes
    x(f,:) = randsample(0:size(selected_features, 1), k);
    x_local(f,:) = x(f,:);
    
    [~, col] = find(x(f,:)~=0);
     x_removed_zeros = x(f, col);
    
    [~, ~, ~, acc_val_x] = MLP(selected_features(x_removed_zeros, :), K_folds, lr, num_epochs, y_train);
    acc_val_x = acc_val_x(end);
    
    if acc_val_x > q
        x_global = x(f,:);
        q = acc_val_x;
    end
    
end


for t = 1:time
    tic
    for f = 1:num_fishes
        [~, col] = find(x(f,:)~=0);
         x_removed_zeros = x(f, col);
        [~, ~, ~, acc_val_x] = MLP(selected_features(x_removed_zeros, :), K_folds, lr, num_epochs, y_train);
        acc_val_x = acc_val_x(end);
        
        [~, col] = find(x_local(f,:)~=0);
        x_local_removed_zeros = x_local(f, col);
        [~, ~, ~, acc_val_x_local] = MLP(selected_features(x_local_removed_zeros, :), K_folds, lr, num_epochs, y_train);
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
        B1 = rand()*a;
        B2 = rand()*a;
        alpha = a/(t^c);
        
        v(f, :) = alpha*v(f, :)+B1*(x_local(f,:)-x(f,:))+B2*(x_global-x(f,:));
        x(f, :) = round(x(f, :) + v(f, :))';
        
        % correct bad x values which cross the limits
        [~, col] = find(x(f, :)>size(selected_features, 1));
        if ~isempty(col)
            x(f, col) = mod(x(f, col), size(selected_features, 1));
        end

        [~, col] = find(x(f, :)<0);
        if ~isempty(col)
            x(f, col) =  abs(x(f, col));
        end
    end
    
    q
    toc
end

%% Genetric Algorithm
clc
close all












