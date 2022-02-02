function [features, num_features] = feature_extraction(X, fs)
    features = zeros(100, size(X, 3), size(X, 2));
    
    for  i = 1:size(X, 3)
        % Mean Freq
        n_f = 1;
        features(n_f,i,:) = meanfreq(X(:,:,i),fs);

        % Med Freq
        n_f = n_f+1;
        features(n_f,i,:) = medfreq(X(:,:,i),fs);

        % Total Power of Channels
        n_f = n_f+1;
        features(n_f,i,:) = bandpower(X(:,:,i));
        
        % Power of Delta Band
        n_f = n_f+1;
        freq_band = [0.5, 4];
        features(n_f,i,:) = bandpower(X(:,:,i), fs, freq_band);

        % Power of Theta Band
        n_f = n_f+1;
        freq_band = [4, 8];
        features(n_f,i,:) = bandpower(X(:,:,i), fs, freq_band);

        % Power of alpha Band
        n_f = n_f+1;
        freq_band = [8, 12];
        features(n_f,i,:) = bandpower(X(:,:,i), fs, freq_band);

        % Power of Beta Band
        n_f = n_f+1;
        freq_band = [12, 35];
        features(n_f,i,:) = bandpower(X(:,:,i), fs, freq_band);

        for j = 1:size(X, 2)
            % Entropy
            n_f_tmp = n_f+1;
            features(n_f_tmp,i,j) = entropy(X(:,j,i));
       
            % Lyapunov Exponent
            n_f_tmp = n_f_tmp+1;
            features(n_f_tmp,i,j) = lyapunovExponent(X(:,j,i), fs);
        
            % Correlation Dim
%             n_f = n_f+1;
%             features(n_f,i,j) = correlationDimension(X(:,j,i));
        end
        
        % Skewness
        n_f = n_f+1;
        features(n_f,i,:) = skewness(X(:,:,i));
        
        % Kurtosis
        n_f = n_f+1;
        features(n_f,i,:) = kurtosis(X(:,:,i));
        
        % OBW
        n_f = n_f+1;
        features(n_f,i,:) = obw(X(:,:,i), fs);
        
    end
    num_features = n_f;
    features = features(1:num_features, :, :);
end