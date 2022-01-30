function features = feature_extraction(X, num_features, fs)
    features = zeros(num_features, size(X, 3), size(X, 2));
    
    for  i = 1:size(X, 3)
        n_f = 1;
        features(n_f,i,:) = meanfreq(X(:,:,i),fs);

        % Med Freq
        n_f = 2;
        features(n_f,i,:) = medfreq(X(:,:,i),fs);

        % Total Power of Channels
        n_f = 3;
        features(n_f,i,:) = bandpower(X(:,:,i));

        % Power of Theta Band
        n_f = 4;
        freq_band = [4, 8];
        features(n_f,i,:) = bandpower(X(:,:,i), fs, freq_band);

        % Power of alpha Band
        n_f = 5;
        freq_band = [8, 12];
        features(n_f,i,:) = bandpower(X(:,:,i), fs, freq_band);

        % Power of Beta Band
        n_f = 6;
        freq_band = [12, 30];
        features(n_f,i,:) = bandpower(X(:,:,i), fs, freq_band);

        % Power of Gamma Band
        n_f = 7;
        freq_band = [30, 50];
        features(n_f,i,:) = bandpower(X(:,:,i), fs, freq_band);

        % Entropy
        n_f = 8;
        features(n_f,i,:) = entropy(X(:,:,i));

        % Lyapunov Exponent
        n_f = 9;
        features(n_f,i,:) = lyapunovExponent(X(:,:,i), fs);

        % Correlation Dim
        n_f = 10;
        features(n_f,i,:) = correlationDimension(X(:,:,i));
    end

end