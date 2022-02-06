function plot_time_freq(X, cls1_indexes, cls2_indexes, fs)

    X_cls1_mean = mean(X(:, :, cls1_indexes), 3);
    X_cls1_mean = mean(X_cls1_mean, 2);
    X_cls2_mean = mean(X(:, :, cls2_indexes), 3);
    X_cls2_mean = mean(X_cls2_mean, 2);
    size(X_cls1_mean)
    subplot(2,1,1)
    plot((1:50)/fs, X_cls1_mean, 'LineWidth', 2)
    title('CLS1 - Avg of all trials and channels in Time Domain')
    subplot(2,1,2)
    plot((1:50)/fs, X_cls2_mean, 'LineWidth', 2)
    title('CLS2 - Avg of all trials and channels in Time Domain')

    figure
    subplot(2,1,1)
    stft(X_cls1_mean,fs,'Window',hamming(10,'periodic'),'OverlapLength',5,'FFTLength',fs);
    title('STFT CLS1 - Avg of all trials and channels in Time Domain')
    subplot(2,1,2)
    stft(X_cls2_mean,fs,'Window',hamming(10,'periodic'),'OverlapLength',5,'FFTLength',fs);
    title('STFT CLS2 - Avg of all trials and channels in Time Domain')