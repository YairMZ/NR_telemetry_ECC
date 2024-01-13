clear
close all
clc

rootdir = '../scripts/results/final_results';
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

%% damaged fields
[tpr,fpr,ppv,acc,bm,f1, tp, fp, rec]  = classifer_metrics(bad_fields,1);
mask = thr>=1e-3 & thr < 1;
bit_flip_mask = zeros(1,20);
bit_flip_mask(1:6:end)=1;
bit_flip_mask= logical(bit_flip_mask);
close all
legend_txt = {};
for idx = 1:6:20
    legend_txt{end+1} = ['$f=' num2str(round(p_vec(idx),3)) '$'];
end
figure
plot(fpr(bit_flip_mask,mask).',tpr(bit_flip_mask,mask).','-o')
xlim([min(fpr(bit_flip_mask,mask),[],'all'), 1])
ylim([0, 1])
xlabel('False positive rate - damaged fields')
ylabel('True positive rate - damaged fields')
legend(legend_txt,'Interpreter','latex')
set(gca,'FontSize',16,'FontName','CMU Serif')
lin.FontSize = 16;
lin.FontName = 'CMU Serif';
exportgraphics(gca,'ROC_curve_damaged_fields.pdf','ContentType','vector')
figure
semilogx(thr(mask),ppv(1:6:20,mask),'-o')
xlabel('$thr_{damaged}$','Interpreter','latex')
ylabel('Precision - damaged fields')
legend(legend_txt,'Location','southwest','Interpreter','latex')
set(gca,'FontSize',16,'FontName','CMU Serif')
lin.FontSize = 16;
lin.FontName = 'CMU Serif';
exportgraphics(gca,'Precision_damaged_fields.pdf','ContentType','vector')

%% valid fields
beta = 0.5;
[tpr,fpr,ppv,acc,bm,f1, tp, fp, rec]  = classifer_metrics(good_fields,beta);
% mask = thr>=1e-2 | thr==0;
mask = thr>=1e-2 & thr<1;
figure
plot(fpr(bit_flip_mask,mask).',tpr(bit_flip_mask,mask).','-o')
xlim([min(fpr(bit_flip_mask,mask),[],'all'), 1])
ylim([0, 1])
xlabel('False positive rate - valid fields')
ylabel('True positive rate - valid fields')
legend(legend_txt,'Interpreter','latex')
set(gca,'FontSize',16,'FontName','CMU Serif')
lin.FontSize = 16;
lin.FontName = 'CMU Serif';
exportgraphics(gca,'ROC_curve_valid_fields.pdf','ContentType','vector')
figure
semilogx(thr(mask),ppv(1:6:20,mask),'-o')
xlabel('$thr_{valid}$','Interpreter','latex')
ylabel('Precision - valid fields')
legend(legend_txt,'Location','southwest','Interpreter','latex')
set(gca,'FontSize',16,'FontName','CMU Serif')
lin.FontSize = 16;
lin.FontName = 'CMU Serif';
exportgraphics(gca,'Precision_valid_fields.pdf','ContentType','vector')