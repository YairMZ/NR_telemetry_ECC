close all
figure;confusionchart(clustering_cm)
title(['Buffer Clustering Algorithm, f=' num2str(p)])
set(gca,'FontSize',16,'FontName','CMU Serif')
filename = ['clustering_' num2str(n_classes) 'classes_' num2str(p) 'p_' num2str(train_with_errors)   'train_error.pdf'];
exportgraphics(gca,filename,'ContentType','vector')


figure;confusionchart(bmm_cm)
title(['Bernoulli Mixture Model, f=' num2str(p)])
set(gca,'FontSize',16,'FontName','CMU Serif')
filename = ['bmm_' num2str(n_classes) 'classes_' num2str(p) 'p_' num2str(train_with_errors)   'train_error.pdf'];
exportgraphics(gca,filename,'ContentType','vector')
