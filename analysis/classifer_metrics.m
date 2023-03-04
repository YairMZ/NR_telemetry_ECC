function [tpr,fpr,ppv,acc,bm,f1, true_p, false_p] = classifer_metrics(fields_performance, beta)

n_thr = length(fields_performance);
tpr = zeros(size(fields_performance{1},1),n_thr);
fpr = tpr;
ppv = tpr;
tnr = tpr;
acc = tpr;
true_p = tpr;
false_p = tpr;
for thr_idx = 1:n_thr
    [tp, fp, tn, ~, p, n, ~] = classifer_confusion(fields_performance{thr_idx});
    tpr(:,thr_idx) = tp./p;
    fpr(:,thr_idx) = fp./n;
    ppv(:, thr_idx) = tp./(tp +fp);
    acc(:,thr_idx) = (tp+tn)./(p+n);
    tnr(:, thr_idx) = tn./n;
    true_p(:,thr_idx) = tp;
    false_p(:,thr_idx) = fp;
end
bm = tpr + tnr - 1;
f1 = (1+beta^2).*tpr.*ppv./(beta^2*tpr + ppv);
end

