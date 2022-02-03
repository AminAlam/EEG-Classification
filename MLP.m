function [loss_train, loss_val, acc_train, acc_val] = MLP(best_features, K_folds, lr, num_epochs, y_train)
    
    k = size(best_features, 1);
    num_datas_in_fold = floor(size(best_features,2)/K_folds);
    num_all_datas = size(best_features,2);
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


        datas_train = py.numpy.array(best_features(:,folds_train)');
        datas_val = py.numpy.array(best_features(:,folds_val)');
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