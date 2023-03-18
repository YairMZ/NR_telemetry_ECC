%% load data
clear
close all
clc
%%
% rootdir = '../scripts/results/cluster_2';
% filelist = dir(fullfile(rootdir, '**/*summary*.mat'));  %get list of files and folders in any subfolder
% filelist = filelist(~[filelist.isdir]);  %remove folders from list
% 
% n_files = length(filelist);
% thr = zeros(1, n_files);
% for idx = 1:n_files
%     load(fullfile(filelist(idx).folder, filelist(idx).name),'args')
%     thr(idx) = args.threshold;
%     filelist(idx).thr = double(args.threshold);
% end
% 
% %sort
% T = struct2table(filelist);
% T = sortrows(T, 'thr');
% filelist = table2struct(T); % change it back to struct array if necessary
% clear T
% 
% % load
% for idx = 1:n_files
%     load(fullfile(filelist(idx).folder, filelist(idx).name))
%     good_fields{idx} = good_fields_performance;
%     bad_fields{idx} = bad_fields_performance;
%     args_arr{idx} = args; 
% end
% 
% 
% n_thr = length(args_arr);
% thr = zeros(1, n_thr);
% for thr_idx = 1:n_thr
%     thr(thr_idx) = args_arr{1,thr_idx}.threshold;
% end

n_p = args.nflips;
p_vec = linspace(args.minflip, args.maxflip, n_p);
n_buffers = results{1,1}.number_of_buffers;

%%
forced_fields = zeros(n_buffers,n_p);
forcedbits = zeros(n_buffers,n_p);
bits_flipped = zeros(n_buffers,n_p);
err_bits_flipped = zeros(n_buffers,n_p);
damaged_fields = zeros(n_buffers,n_p);

for ii = 1:n_p
    class_res = results{1,ii}.classifier;
    for jj = 1:n_buffers
        forced_fields(jj,ii) = length(class_res.forced_fields{jj});
        forcedbits(jj,ii) = length(class_res.forced_bits{jj});
        bits_flipped(jj,ii) = class_res.n_bits_flipped(jj);
        err_bits_flipped(jj,ii) = class_res.erroneously_flipped_bits(jj);
        damaged_fields(jj,ii) = length(class_res.damaged_fields{jj});
    end
end
bad_forcing_rate = forced_fields./damaged_fields;