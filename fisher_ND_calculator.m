function best_features = fisher_ND_calculator(k, num_itters, selected_features, cls1_indexes, cls2_indexes)

    selected_samples_memory = zeros(num_itters, k);
    FS_ND = zeros(1,100);
    for itter = 1:num_itters
       samples_feature_indexes = randsample(1:size(selected_features, 1), k);
       selected_samples_memory(itter, :) = samples_feature_indexes;
       sigg_cls1 = selected_features(samples_feature_indexes, cls1_indexes);
       sigg_cls2 = selected_features(samples_feature_indexes, cls2_indexes);
       FS_ND(1, itter) = fisher_score_ND(sigg_cls1, sigg_cls2, selected_features(samples_feature_indexes, :));
    end
    max(FS_ND)
    [~, col] = find(FS_ND == max(FS_ND));
    best_features_indexes = selected_samples_memory(col(1), :);
    best_features = selected_features(best_features_indexes,:);