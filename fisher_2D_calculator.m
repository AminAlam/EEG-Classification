function [selected_features, row, col] = fisher_2D_calculator(num_features, features, cls1_indexes, cls2_indexes)
    FS = zeros(num_features, size(features,3));
    for ch = 1:size(features,3)
        for f = 1:num_features
           sigg_cls1 = features(f, cls1_indexes, ch);
           sigg_cls2 = features(f, cls2_indexes, ch);
           FS(f, ch) = fisher_score_2D(sigg_cls1, sigg_cls2, features(f, :, ch));
        end
    end

    mean_fisher = mean(FS, 'all');
    thresh = mean_fisher;
    [row, col] = find(FS>thresh);
    selected_features = zeros(length(row), size(features,2));
    for i = 1:length(row)
        selected_features(i,:) = features(row(i), :, col(i));
    end
    selected_features = cell2mat(arrayfun(@(x) features(row(x), :, col(x)), 1:length(row),'UniformOutput',false)');
