function J = fisher_score_2D(sigg_sub1, sigg_sub2, all_sub)
    u0 = mean(all_sub,'all');
    u1 = mean(sigg_sub1,'all');
    u2 = mean(sigg_sub2,'all');
    var1 = var(sigg_sub1);
    var2 = var(sigg_sub2);
    J = (abs(u0-u1)^2+abs(u0-u2)^2)/(var1+var2);