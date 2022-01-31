function J = fisher_score_ND(sigg_cls1, sigg_cls2, all_sub)

    mu_0 = mean(all_sub')';
    mu_1 = mean(sigg_cls1')';
    mu_2 = mean(sigg_cls2')';
    S1 = 0;
    S2 = 0;

    for sub1 = 1:size(sigg_cls1, 2)
        S1 = S1+(sigg_cls1(:,sub1)-mu_1)*(sigg_cls1(:,sub1)-mu_1)';
    end
    S1 = S1/sub1;

    for sub2 = 1:size(sigg_cls2, 2)
        S2 = (sigg_cls2(:,sub2)-mu_2)*(sigg_cls2(:,sub2)-mu_2)';
    end
    S2 = S2/sub2;

    Sw = S1+S2;
    Sb = ((mu_1-mu_0)*(mu_1-mu_0)'+(mu_2-mu_0)*(mu_2-mu_0)');

    J = trace(Sb)/trace(Sw);