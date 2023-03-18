function res = recall(classifer_performance)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
[tp, fp, tn, fn, p, n, cm] = classifer_confusion(classifer_performance);
res = tp./(tp +fn);
end

