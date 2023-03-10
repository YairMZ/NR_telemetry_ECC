function [tp, fp, tn, fn, p, n, cm] = classifer_confusion(fields_performance)
% first row/column valid (positive), second row/column damaged (negative).
fields_performance = double(fields_performance);
tp = fields_performance(:,1);
fp = fields_performance(:,2);
tn = fields_performance(:,3);
fn = fields_performance(:,4);
p = fields_performance(:,5);
n = fields_performance(:,6);
for idx = 1:size(fields_performance,1)
    cm{idx} = [[tp(idx), fn(idx)];[fp(idx), tn(idx)]];
end
% figure
% confusionchart(good_fields_confusion,["valid field","possibly damaged field"])
end