function features = feature_extraction(X, num_features, fs)
    features = zeros(num_features, size(X, 3), size(X, 2));
    
    for  i = 1:size(X, 3)
        % Mean Freq
        n_f = 1;
        features(n_f,i,:) = meanfreq(X(:,:,i),fs);

        % Med Freq
        n_f = 2;
        features(n_f,i,:) = medfreq(X(:,:,i),fs);

        % Total Power of Channels
        n_f = 3;
        features(n_f,i,:) = bandpower(X(:,:,i));
        
        % Power of Delta Band
        n_f = 4;
        freq_band = [0.5, 4];
        features(n_f,i,:) = bandpower(X(:,:,i), fs, freq_band);

        % Power of Theta Band
        n_f = 5;
        freq_band = [4, 8];
        features(n_f,i,:) = bandpower(X(:,:,i), fs, freq_band);

        % Power of alpha Band
        n_f = 6;
        freq_band = [8, 12];
        features(n_f,i,:) = bandpower(X(:,:,i), fs, freq_band);

        % Power of Beta Band
        n_f = 7;
        freq_band = [12, 35];
        features(n_f,i,:) = bandpower(X(:,:,i), fs, freq_band);

        for j = 1:size(X, 2)
            % Entropy
            n_f = 8;
            features(n_f,i,j) = entropy(X(:,j,i));
       
            % Lyapunov Exponent
            n_f = 9;
            features(n_f,i,j) = lyapunovExponent(X(:,j,i), fs);
        
            % Correlation Dim
            n_f = 10;
            features(n_f,i,j) = correlationDimension(X(:,j,i));
        end
        
        % Skewness
        n_f = 11;
        features(n_f,i,:) = skewness(X(:,:,i));
        
        % Kurtosis
        n_f = 12;
        features(n_f,i,:) = kurtosis(X(:,:,i));
        
        % OBW
        n_f = 13;
        features(n_f,i,:) = obw(X(:,:,i), fs);
        
    end

end