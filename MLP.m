function [mean_loss_train, mean_loss_val, mean_acc_train, mean_acc_val] = MLP(datas, K_folds, lr, num_epochs, y_train, k, fs, num_itters)
    
    num_datas_in_fold = floor(size(datas, 3)/K_folds);
    num_all_datas = size(datas, 3);
    loss_train = [];
    loss_val = [];
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

        datas_train = datas(:, :, folds_train);
        datas_val = datas(:, :, folds_val);
        
        W_Filt = CSP(datas_train, y_train(:,folds_train));
        
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
        
        [~, cls1_indexes] = find(y_train(folds_train)==0);
        [~, cls2_indexes] = find(y_train(folds_train)==1);
        
        [selected_features, row_selected, col_selected] = fisher_2D_calculator(num_features, features_train, cls1_indexes, cls2_indexes);
        
        
        [best_features_train, best_features_indexes] = fisher_ND_calculator(k, num_itters, selected_features, cls1_indexes, cls2_indexes);
        
        features_index_col = col_selected(best_features_indexes);
        features_index_row = row_selected(best_features_indexes);
        best_features_val = zeros(length(features_index_row), size(datas_val, 3));
        for i = 1:length(features_index_row)
            best_features_val(i, :) = features_val(features_index_row(i), :, features_index_col(i));
        end
        
        
        
        datas_train = py.numpy.array(best_features_train');
        datas_val = py.numpy.array(best_features_val');
        labels_train = py.numpy.array(y_train(:,folds_train)');
        labels_val = py.numpy.array(y_train(:,folds_val)');

        kwa = pyargs('datas_train', datas_train, ...
            'datas_val', datas_val, ...
            'labels_train', labels_train, ...
            'labels_val', labels_val, ...
            'input_size', int32(k), ...
            'ouput_size', int32(1), ...
            'lr', double(lr), ...
            'num_epochs', int32(num_epochs));

        mod = py.importlib.import_module('classifier');
        py.importlib.reload(mod);
        out = mod.call_from_matlab(kwa);

        loss_train = [loss_train; str2num(char(out.cell{1}))];
        loss_val = [loss_val; str2num(char(out.cell{2}))];
        acc_train = [acc_train; str2num(char(out.cell{3}))];
        acc_val = [acc_val; str2num(char(out.cell{4}))];

    end

    mean_loss_train = mean(loss_train, 1);
    mean_loss_val = mean(loss_val, 1);
    mean_acc_train = mean(acc_train,1);
    mean_acc_val = mean(acc_val,1);