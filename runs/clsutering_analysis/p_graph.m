clear
%%
% Specify the directory you want to search
folder = 'results/p_graph/';

% Recursively search for .mat files in the directory and its subfolders
file_list = dir(fullfile(folder, '**', '*.mat'));

for i = 1:length(file_list)
    file_path = fullfile(file_list(i).folder, file_list(i).name);
    data = load(file_path);
    file_list(i).p = data.p;
end

%sort
T = struct2table(file_list);
T = sortrows(T, 'p');
file_list = table2struct(T); % change it back to struct array if necessary
clear T

bmm_acc_vec = zeros(1,length(file_list));
cls_acc_vec = zeros(1,length(file_list));
p_vec = zeros(1,length(file_list));
for i = 1:length(file_list)
    file_path = fullfile(file_list(i).folder, file_list(i).name);
    data = load(file_path);
    bmm_acc_vec(i) = data.bmm_accuracy;
    cls_acc_vec(i) = data.clustering_accuracy;
    p_vec(i) = data.p;
end
clear data

%%
figure
plot(p_vec,bmm_acc_vec, p_vec,cls_acc_vec)