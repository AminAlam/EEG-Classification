function [mean_loss_train, mean_loss_val, mean_acc_train, mean_acc_val] = MLP_PSO(best_features_train, best_features_val, y_data_train, y_data_val, lr, num_epochs)
    
    loss_train = [];
    loss_val = [];
    acc_train = [];
    acc_val = [];
    
    k = size(best_features_train, 1);


    datas_train = py.numpy.array(best_features_train');
    datas_val = py.numpy.array(best_features_val');
    labels_train = py.numpy.array(y_data_train);
    labels_val = py.numpy.array(y_data_val);

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



    mean_loss_train = mean(loss_train, 1);
    mean_loss_val = mean(loss_val, 1);
    mean_acc_train = mean(acc_train,1);
    mean_acc_val = mean(acc_val,1);