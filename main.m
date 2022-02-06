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
W_Filt = CSP(X, y_train);
X_filt = zeros(size(X, 1), size(W_Filt, 2), size(X, 3));

for i = 1:size(X, 3)
    X_filt(:,:,i) = (W_Filt'*X(:,:,i)')';
end

plot_time_freq(X_filt, cls1_indexes, cls2_indexes, fs)
X = X_filt;
%% Feature Extraction
clc
close all
[features, num_features] = feature_extraction(X, fs);
%% plot features
clc
close all
f_index = 6;
scatter(mean(features(f_index,cls1_indexes,:), 3), cls1_indexes*0, 'r')
hold on
scatter(mean(features(f_index,cls2_indexes,:), 3), cls2_indexes*0+1, 'b')
%% Calculating the 2D Fisher Score
clc
[selected_features, row_selected, col_selected] = fisher_2D_calculator(num_features, features, cls1_indexes, cls2_indexes);
%% Calculating the ND Fisher Score
clc
k = 6;
num_itters = 100;
[best_features, best_features_indexes] = fisher_ND_calculator(k, num_itters, selected_features, cls1_indexes, cls2_indexes);% MLP
%%
clc
close all
k = 16;
num_itters = 100;
K_folds = 5;
lr = 0.05;
num_epochs = 50;

[loss_train, loss_val, acc_train, acc_val] = MLP(X, K_folds, lr, num_epochs, y_train, k, fs, num_itters);


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

lr = 0.01;
num_epochs = 50;
num_fishes = 200;
time = 50;
a = 4;
c = 0.5;
k = 32;
v = zeros(num_fishes, k);
x = zeros(num_fishes, k);

datas_train = x_train(:, :, 1:250);
y_data_train = y_train(1, 1:250);
datas_val = x_train(:, :, 251:end);
y_data_val = y_train(1, 251:end);

W_Filt = CSP(datas_train, y_data_train);

datas_train_filt = zeros(size(datas_train, 1), size(W_Filt, 2), size(datas_train, 3));
datas_val_filt = zeros(size(datas_val, 1), size(W_Filt, 2), size(datas_val, 3));

for i = 1:size(datas_train, 3)
    datas_train_filt(:,:,i) = (W_Filt'*datas_train(:,:,i)')';
end

for i = 1:size(datas_val, 3)
    datas_val_filt(:,:,i) = (W_Filt'*datas_val(:,:,i)')';
end


[features_train, num_features] = feature_extraction(datas_train_filt, fs);
[features_val, num_features] = feature_extraction(datas_val_filt, fs);

features_train = permute(features_train, [1, 3, 2]);
features_train = reshape(features_train, [size(features_train, 1)*size(features_train, 2) , size(features_train, 3)]);
features_val = permute(features_val, [1, 3, 2]);
features_val = reshape(features_val, [size(features_val, 1)*size(features_val, 2) , size(features_val, 3)]);

max_feature_index = size(features_train, 1);
x_local = zeros(num_fishes, k);
x_global = zeros(1,k);
q = -inf;

for f = 1:num_fishes
    x(f,:) = randsample(0:max_feature_index, k);
    x_local(f,:) = x(f,:);
    
    [~, col] = find(x(f,:)~=0);
    x_removed_zeros = x(f, col);
    
    best_features_train = features_train(x_removed_zeros, :);
    best_features_val = features_val(x_removed_zeros, :);
    
    [~, ~, ~, acc_val_x] = MLP_PSO(best_features_train, best_features_val, y_data_train, y_data_val, lr, num_epochs);
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
        best_features_train = features_train(x_removed_zeros, :);
        best_features_val = features_val(x_removed_zeros, :);
        [~, ~, ~, acc_val_x] = MLP_PSO(best_features_train, best_features_val, y_data_train, y_data_val, lr, num_epochs);
        acc_val_x = acc_val_x(end);
        
        [~, col] = find(x_local(f,:)~=0);
        x_local_removed_zeros = x_local(f, col);
        best_features_train = features_train(x_local_removed_zeros, :);
        best_features_val = features_val(x_local_removed_zeros, :);
        [~, ~, ~, acc_val_x_local] = MLP_PSO(best_features_train, best_features_val, y_data_train, y_data_val, lr, num_epochs);
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
        [~, col] = find(x(f, :)>max_feature_index);
        if ~isempty(col)
            x(f, col) = max_feature_index;
        end

        [~, col] = find(x(f, :)<0);
        if ~isempty(col)
            x(f, col) =  0;
        end
    end
    
    q
    toc
end

%% Evaluation
clc

% checking accuracy of network
X_train = x_train(:,:,1:250);
Y_train = y_train(1:250);

X_val = x_train(:,:,251:end);
Y_val = y_train(251:end);

W_Filt = CSP(X_train, Y_train);

X_train_filt = zeros(size(X_train, 1), size(W_Filt, 2), size(X_train, 3));
for i = 1:size(X_train, 3)
    X_train_filt(:,:,i) = (W_Filt'*X_train(:,:,i)')';
end

X_val_filt = zeros(size(X_val, 1), size(W_Filt, 2), size(X_val, 3));
for i = 1:size(X_val, 3)
    X_val_filt(:,:,i) = (W_Filt'*X_val(:,:,i)')';
end

X_val = X_val_filt;
X_train = X_train_filt;

[features_train, ~] = feature_extraction(X_train, fs);
[features_val, ~] = feature_extraction(X_val, fs);

features_train = permute(features_train, [1, 3, 2]);
features_train = reshape(features_train, [size(features_train, 1)*size(features_train, 2) , size(features_train, 3)]);
features_val = permute(features_val, [1, 3, 2]);
features_val = reshape(features_val, [size(features_val, 1)*size(features_val, 2) , size(features_val, 3)]);

[row, col] = find(x_global~=0);
x_global_without_zeros = x_global(1, col);


features2model_train = zeros(size(X_train, 3), length(x_global_without_zeros));
features2model_val = zeros(size(X_val, 3), length(x_global_without_zeros));

for i = 1:length(x_global_without_zeros)
    features2model_train(:, i) = features_train(x_global_without_zeros(i),:);
    features2model_val(:, i) = features_val(x_global_without_zeros(i),:);
end


clc
datas_train = py.numpy.array(features2model_train);
datas_val = py.numpy.array(features2model_val);
labels_train = py.numpy.array(Y_train');
labels_val = py.numpy.array(Y_val');

kwa = pyargs('datas_train', datas_train, ...
    'datas_val', datas_val, ...
    'labels_train', labels_train, ...
    'labels_val', labels_val, ...
    'input_size', int32(length(x_global_without_zeros)), ...
    'ouput_size', int32(1), ...
    'lr', double(lr), ...
    'num_epochs', int32(num_epochs), ...
    'save_model', int32(1));

mod = py.importlib.import_module('classifier');
py.importlib.reload(mod);
out = mod.call_from_matlab(kwa);

loss_train = str2num(char(out.cell{1}));
loss_val = str2num(char(out.cell{2}));
acc_train = str2num(char(out.cell{3}));
acc_val = str2num(char(out.cell{4}));
acc_train(end)
acc_val(end)

%%
clc
x_test_filt = zeros(size(x_test, 1), size(W_Filt, 2), size(x_test, 3));
for i = 1:size(x_test, 3)
    x_test_filt(:, :, i) = (W_Filt'*x_test(:,:,i)')';
end

[features_test, num_features] = feature_extraction(x_test_filt, fs);

features_test = permute(features_test, [1, 3, 2]);
features_test = reshape(features_test, [size(features_test, 1)*size(features_test, 2) , size(features_test, 3)]);

features2model = zeros(size(x_test, 3), length(x_global_without_zeros));

for i = 1:length(x_global_without_zeros)
    features2model(:, i) = features_test(x_global_without_zeros(i),:);
end

datas = py.numpy.array(features2model);


kwa = pyargs('datas', datas, ...
    'input_size', int32(length(x_global_without_zeros)), ...
    'ouput_size', int32(1));

mod = py.importlib.import_module('classifier');
py.importlib.reload(mod);
out = mod.Eval(kwa);

labels_test = [];
for i = 1:100
    labels_test = [labels_test, str2num(char(out.cell{i}))];
end

test_labels_true = [ 0     0     0     0     1     0     0     0     1     1 1     0     0     1     1     0     0     1     0     0 0     0     0     1     0     1     1     1     0     1 0     0     0     1     0     0     1     0     1     0 1     1     0     0     0     1     0     0     0     1 1     1     1     0     0     1     1     1     0     1 1     0     0     1     0     0     0     1     0     1 0     1     1     0     0     0     0     1     0     1 0     1     1     0     1     1     0     0     1     1 0     0     1     0     1     1     0     1     1     0];







