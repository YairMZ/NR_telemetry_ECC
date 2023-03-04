% first row/column valid (positive), second row/column damaged (negative).
% [[true positive | ];
tmp = sum(good_fields_performance,1);
tp = tmp(1); fp = tmp(2); tn = tmp(3); fn = tmp(4); p = tmp(5); n = tmp(6);
good_fields_confusion = [[tp, fn];[fp, tn]];