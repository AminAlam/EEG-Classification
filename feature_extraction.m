function [features, num_features] = feature_extraction(X, fs)
    features = zeros(100, size(X, 3), size(X, 2));
    
    for  i = 1:size(X, 3)
        % 1 Mean Freq
        n_f = 1;
        features(n_f,i,:) = meanfreq(X(:,:,i),fs);

        % 2 Med Freq
        n_f = n_f+1;
        features(n_f,i,:) = medfreq(X(:,:,i),fs);

        % 3 Total Power of Channels
        n_f = n_f+1;
        features(n_f,i,:) = bandpower(X(:,:,i));

        % 4 Power of 8-12
        n_f = n_f+1;
        freq_band = [8, 12];
        features(n_f,i,:) = bandpower(X(:,:,i), fs, freq_band);

        % 5 Power of 12-16
        n_f = n_f+1;
        freq_band = [12, 16];
        features(n_f,i,:) = bandpower(X(:,:,i), fs, freq_band);

        % 6 Power of 16-20
        n_f = n_f+1;
        freq_band = [16, 20];
        features(n_f,i,:) = bandpower(X(:,:,i), fs, freq_band);

        % 7 Power of 20-24
        n_f = n_f+1;
        freq_band = [20, 24];
        features(n_f,i,:) = bandpower(X(:,:,i), fs, freq_band);

        for j = 1:size(X, 2)
            % 8 Entropy
            n_f_tmp = n_f+1;
            features(n_f_tmp,i,j) = entropy(X(:,j,i));

            % 9 Wavelet Packet Shannon Entropy
            n_f_tmp = n_f_tmp+1;
            features(n_f_tmp,i,j) = wentropy(X(:,j,i),'shannon');

            % 10 Wavelet Packet Log Energy Entropy
            n_f_tmp = n_f_tmp+1;
            features(n_f_tmp,i,j) = wentropy(X(:,j,i),'log energy');

            % 11 Lyapunov Exponent
            n_f_tmp = n_f_tmp+1;
            features(n_f_tmp,i,j) = lyapunovExponent(X(:,j,i), fs);

            % Correlation Dim
            n_f_tmp = n_f_tmp+1;
            features(n_f_tmp,i,j) = correlationDimension(X(:,j,i));
            
            % mean of diff of trial
            n_f_tmp = n_f_tmp+1;
            features(n_f_tmp,i,j) = mean(diff(X(:,j,i)));
        end

        % 12 Skewness
        n_f = n_f_tmp+1;
        features(n_f,i,:) = skewness(X(:,:,i));

        % 13 Kurtosis
        n_f = n_f+1;
        features(n_f,i,:) = kurtosis(X(:,:,i));

        % 14 OBW
        n_f = n_f+1;
        features(n_f,i,:) = obw(X(:,:,i), fs);
        
    end
    num_features = n_f;
    features = features(1:num_features, :, :);
end