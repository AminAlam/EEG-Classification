function [acc_train, acc_val] = RBF(K_folds, best_features, y_train)


    acc_train = [];
    acc_val = [];
    
    k = size(best_features, 1);
    num_datas_in_fold = floor(size(best_features,2)/K_folds);
    num_all_datas = size(best_features,2);


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