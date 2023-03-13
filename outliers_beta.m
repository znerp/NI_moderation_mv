% Optional 1st step of the multivariate moderation model:

% the script identifies for every participant in how many voxels its beta 
% coefficient would be considered an extreme outlier (based on 2nd quartile
% - 3*IQR or 3rd quartile + 3*IQR) in the distribution of all participants'
% beta values
% saves this number (+ the percentage) in the table for every participant 
% for later usage


clear; close all
%%
mod = 'dm'; % contrast, e.g. 'dm'

path_table = ['../../data/m0/tables/multivariate/covs_arcsin_CSF_s6_sub493_' mod '.csv']; % table with all subjects
path_func = '../../data/m0/derivatives/glm/contrasts/arcsin_CSF/%s_%s.nii'; % modality and subject ID are inserted below
mask = ['../../data/masks/mask_GM35_shoot_0.2_noNaN' mod '.nii'];

percent = 10; % how many percent voxels with extreme outliers are maximally tolerated

save = 0 % append number of voxels with extreme outliers to table?
%%
% choose subset
T = readtable(path_table);
N = height(T);

fnames = cell(N,1);
for ix=1:N
    fnames{ix} = sprintf(path_func, mod, T.ID{ix});
end

files = spm_vol(char(fnames));

Y = spm_summarise(files,mask);

%% get a feeling for statistics on the different voxels
nvox = size(Y,2);
stats = struct('min', nan(nvox,1), 'max', nan(nvox,1), 'mean', nan(nvox,1), 'median', nan(nvox,1));
for ivox=1:nvox
    stats.min(ivox)= min(Y(:,ivox));
    stats.max(ivox)= max(Y(:,ivox));
    stats.median(ivox)= median(Y(:,ivox));
    stats.mean(ivox)= mean(Y(:,ivox));
end


% figure;
% histogram(stats.min, 20)
% title('min')
% 
% figure;
% histogram(stats.max, 20)
% title('max')
% 
% figure;
% histogram(stats.mean, 20)
% title('mean')
% 
% figure;
% histogram(stats.median, 20)
% title('median')


%% identify for every subject how many extreme outliers they have
n_out = zeros(N,1);
for ivox=1:nvox
    % identify extreme outliers
    idc_out = find(isoutlier_IQR(Y(:,ivox),3));
    n_out(idc_out) = n_out(idc_out)+1;
end

[n_out_sorted, idx_ID] = sort(n_out, 'descend');

ID_most_out = T.ID(idx_ID);


N = sum(n_out_sorted > nvox*percent/100); % number of participants with outlier values in more than the previously specified percentage of the voxels

fprintf('\nOutliers in up to %d %% of the voxels were tolerated.', percent)
for ii=1:N
    fprintf('\n%d. %s has %d outliers (%.2f %%)', ii, ID_most_out{ii}, n_out_sorted(ii), n_out_sorted(ii)/nvox*100)
end
fprintf('\n')

%% also save the number of extreme outliers to the table
T.(['n_outliers_extreme_' mod]) = n_out;
T.(['perc_outliers_extreme_' mod]) = n_out / nvox * 100;

if save 
    writetable(T, path_table)
end
