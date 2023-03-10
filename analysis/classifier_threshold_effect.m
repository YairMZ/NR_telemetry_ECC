clear
close all
clc

rootdir = '../scripts/results/cluster_1';
filelist = dir(fullfile(rootdir, '**/*summary*.mat'));  %get list of files and folders in any subfolder
filelist = filelist(~[filelist.isdir]);  %remove folders from list

n_files = length(filelist);
thr = zeros(1, n_files);
for idx = 1:n_files
    load(fullfile(filelist(idx).folder, filelist(idx).name),'args')
    thr(idx) = args.threshold;
    filelist(idx).thr = double(args.threshold);
end

%sort
T = struct2table(filelist);
T = sortrows(T, 'thr');
filelist = table2struct(T); % change it back to struct array if necessary
clear T



%%
% load
for idx = 1:n_files
    load(fullfile(filelist(idx).folder, filelist(idx).name))
    good_fields{idx} = good_fields_performance;
    bad_fields{idx} = bad_fields_performance;
    args_arr{idx} = args; 
end


n_thr = length(args_arr);
thr = zeros(1, n_thr);
for thr_idx = 1:n_thr
    thr(thr_idx) = args_arr{1,thr_idx}.threshold;
end

n_p = size(bad_fields{1,1},1);
p_vec = linspace(args_arr{1,1}.minflip, args_arr{1,1}.maxflip, n_p);
clear args bad_fields_performance good_fields_performance args_arr

%%
[tpr,fpr,ppv,acc,bm,f1, tp, fp, rec]  = classifer_metrics(bad_fields,1);
% mask = thr>=1e-3 | thr==0;
mask = thr>=1e-3 & thr < 1;
close all
legend_txt = {};
for idx = 1:6:20
    legend_txt{end+1} = ['f=' num2str(p_vec(idx))];
end
figure
subplot(2,2,1)
plot(fpr(end-2:end,mask).',tpr(end-2:end,mask).','-o')
xlim([0, 1])
ylim([0, 1])
xlabel('False positive rate - bad fields')
ylabel('True positive rate - bad fields')
title('ROC')

subplot(2,2,2)
plot(thr(mask),tpr(end-2:end,mask).','-o')
xlabel('Threshold')
ylabel('True positive rate - bad fields')

subplot(2,2,3)
plot(thr(mask),fpr(end-2:end,mask).','-o')
xlabel('Threshold')
ylabel('False positive rate - bad fields')

div = max(tp(1:6:20,mask)+fp(1:6:20,mask),[],'all');
subplot(2,2,4)
plot(tp(1:6:20,mask).'/div+fp(1:6:20,mask).'/div,tp(1:6:20,mask).'/div,[0,1],[0,1] )
xlabel('TP +FP')
ylabel('TP')
title('TOC')
legend({legend_txt{:}, 'linear reference'})

figure
subplot(2,3,1)
plot(thr(mask),ppv(1:6:20,mask))
title('PPV - bad fields')
xlabel('Treshold')
ylabel('Precision')
legend(legend_txt)

subplot(2,3,2)
plot(thr(mask),rec(1:6:20,mask))
title('recall - bad fields')
xlabel('Treshold')
ylabel('Recall')
legend(legend_txt)


tmp = zeros(length(p_vec),6);
for idx = 1:length(thr)
    tmp = tmp + double(bad_fields{idx});
end
tmp = tmp(:,end-1:end)/length(thr);
%positive/(positive + negative)
baseline = tmp(:,1)./sum(tmp,2);
subplot(2,3,3)
plot(rec(1:6:20,mask).',ppv(1:6:20,mask).')
yline(baseline(1),'-','baseline');
yline(baseline(19),'-','baseline');
title('PR')
xlabel('Recall')
ylabel('Precision')
legend(legend_txt)

subplot(2,3,4)
plot(thr(mask),acc(1:6:20,mask))
title('ACC')
xlim([1e-3,1])
ylim([0.65 max(ylim)])
legend(legend_txt)

subplot(2,3,5)
plot(thr(mask),bm(1:6:20,mask))
title('BM - Informedness')
xlim([1e-3,1])
ylim([0.3 max(ylim)])
legend(legend_txt)

subplot(2,3,6)
plot(thr(mask),f1(1:6:20,mask))
title('F1')
xlim([1e-2,1])
legend(legend_txt)
%%
beta = 0.5;
[tpr,fpr,ppv,acc,bm,f1, tp, fp, rec]  = classifer_metrics(good_fields,beta);
% mask = thr>=1e-2 | thr==0;
mask = thr>=1e-2 & thr<1;
figure
subplot(2,2,1)
plot(fpr(end-2:end,mask).',tpr(end-2:end,mask).','-o')
xlim([0, 1])
ylim([0, 1])
xlabel('False positive rate - good fields')
ylabel('True positive rate - good fields')
title('ROC')

subplot(2,2,2)
plot(thr(mask),tpr(end-2:end,mask).','-o')
xlabel('Threshold')
ylabel('True positive rate - good fields')

subplot(2,2,3)
plot(thr(mask),fpr(end-2:end,mask).','-o')
xlabel('Threshold')
ylabel('False positive rate - good fields')

div = max(tp(1:6:20,mask)+fp(1:6:20,mask),[],'all');
subplot(2,2,4)
plot(tp(1:6:20,mask).'/div+fp(1:6:20,mask).'/div,tp(1:6:20,mask).'/div,[0,1],[0,1] )
xlabel('TP +FP')
ylabel('TP')
title('TOC')
legend({legend_txt{:}, 'linear reference'})

figure
subplot(2,3,1)
plot(thr(mask),ppv(1:6:20,mask))
title('PPV - good fields')
xlabel('Treshold')
ylabel('Precision')
legend(legend_txt)

subplot(2,3,2)
plot(thr(mask),rec(1:6:20,mask))
title('recall - bad fields')
xlabel('Treshold')
ylabel('Recall')
legend(legend_txt)

tmp = zeros(length(p_vec),6);
for idx = 1:length(thr)
    tmp = tmp + double(good_fields{idx});
end
tmp = tmp(:,end-1:end)/length(thr);
%positive/(positive + negative)
baseline = tmp(:,1)./sum(tmp,2);
subplot(2,3,3)
plot(rec(1:6:20,mask).',ppv(1:6:20,mask).')
yline(baseline(19),'-','baseline');
yline(baseline(1),'-','baseline');
title('PR')
xlabel('Recall')
ylabel('Precision')
legend(legend_txt)

subplot(2,3,4)
plot(thr(mask),acc(1:6:20,mask))
title('ACC')
xlim([1e-2,1])
ylim([0.65 max(ylim)])
legend(legend_txt)

subplot(2,3,5)
plot(thr(mask),bm(1:6:20,mask))
title('BM - Informedness')
xlim([1e-2,1])
ylim([0.3 max(ylim)])
legend(legend_txt)

subplot(2,3,6)
plot(thr(mask),f1(1:6:20,mask))
title(['F' num2str(beta)])
xlim([1e-2,1])
legend(legend_txt)