clear
close all
clc

rootdir = '../scripts/results/test';
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
[tpr,fpr,ppv,acc,bm,f1, tp, fp]  = classifer_metrics(bad_fields,1);
mask = thr>=1e-3 | thr==0;
close all
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

figure
subplot(2,2,1)
plot(thr(mask),ppv(:,mask))
title('PPV - bad fields')
xlabel('Treshold')
ylabel('Precision')

subplot(2,2,2)
plot(thr(mask),acc(1:6:20,mask))
title('ACC')
xlim([1e-3,1])
ylim([0.65 max(ylim)])

subplot(2,2,3)
plot(thr(mask),bm(:,mask))
title('BM - Informedness')
xlim([1e-3,1])
ylim([0.3 max(ylim)])

subplot(2,2,4)
plot(thr(mask),f1(:,mask))
title('F1')
xlim([1e-2,1])
%%
beta = 0.5;
[tpr,fpr,ppv,acc,bm,f1, tp, fp]  = classifer_metrics(good_fields,beta);
mask = thr>=1e-2 | thr==0;
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

figure
subplot(2,2,1)
plot(thr(mask),ppv(:,mask))
title('PPV - good fields')
xlabel('Treshold')
ylabel('Precision')

subplot(2,2,2)
plot(thr(mask),acc(1:6:20,mask))
title('ACC')
xlim([1e-2,1])
ylim([0.65 max(ylim)])

subplot(2,2,3)
plot(thr(mask),bm(:,mask))
title('BM - Informedness')
xlim([1e-2,1])
ylim([0.3 max(ylim)])

subplot(2,2,4)
plot(thr(mask),f1(:,mask))
title(['F' num2str(beta)])
xlim([1e-2,1])