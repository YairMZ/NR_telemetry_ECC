clear

rootdir = '../scripts/results/cluster_1';
filelist = dir(fullfile(rootdir, '**/*.mat'));  %get list of files and folders in any subfolder
filelist = filelist(~[filelist.isdir]);  %remove folders from list

%sort
T = struct2table(filelist);
T = sortrows(T, 'datenum');
filelist = table2struct(T); % change it back to struct array if necessary
clear T

% load
n_files = length(filelist);
for idx = 1:n_files
    load(fullfile(filelist(idx).folder, filelist(idx).name))
    res_arr{idx} = results;
end
clear results

%%
thr = 0.01:0.01:0.09;
n_thr = length(res_arr);
n_p = length(res_arr{1,1});
fpr_good_fields = zeros(n_p, n_thr);
for p_idx = 1:np
    p = res_arr{1,1}{1,p_idx}.p;
    for thr_idx = 1:n_thr
        % for good fields, negatives are damaged fields
        negatives_per_buffer = res_arr{1,thr_idx}{1,p_idx}.classifier.damaged_fields;
        negatives = 0;
        for idx = 1:length(negatives_per_buffer)
            negatives = negatives + length(negatives_per_buffer{idx});
        end
    end
end