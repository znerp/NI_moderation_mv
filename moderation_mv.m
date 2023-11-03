% MAIN SCRIPT 
% 3rd step of the multivariate moderation model framework: 
% runs the actual multivariate moderation model presented below
% with the selected number of PCs

% The idea here is to reduce the fMRI contrast in the chosen voxels to a 
% number of principal components and use them in a multivariate regression. 
% The regression weights can then be mapped back onto the voxels to 
% determine their individual contributions. 

% moderation_mv_cv_metaloop.m should be run first in order to determine the
% optimal number of principal components entered below. 

% If you want to exclude participants based on the amount of voxels in
% which the beta values from their contrast images are considered extreme
% outliers in the context of the whole sample, run outliers_beta.m before.

% No inference is made about the significance of a single voxel's
% moderation coefficient (=CR contribution). If this is desired, the script
% moderation_mv_bt should be run additionally.


%%% multivariate moderation model:
% cognition = b0 + b1*DP + b2_1*fmri_PC1 + b3_1*DP*fmri_PC1 + b2_2*fmri_PC2 + 
% b3_2*DP*fmri_PC2 + ... + b2_n*fmri_PC1 + b3_n*DP*fmri_PC1 + C*cov

% DV: cognition, e.g. pacc5
% IVs:  DP score (later named PL), 
%       principal component of DM contrast,
%       interaction between DP and dm_PC
% covs: age
%       sex
%       TIV
%       sites



%% initialize variables
clear; close all hidden

n_pc = 7; % number of principal components (determined with moderation_mv_cv_metaloop.m)
use_covs = 1; % should covariates be used?
save_tbl = 0; % should the CR scores be appended to the provided table? 
save_pc_imgs = 0; % should the weights of the PCs be saved in an image?

% exclusion of extreme outliers based on their beta values; requires
% outliers_beta.m to be run before
remove_outliers = 1
thresh_outl = 10; % percent of voxels with extreme outliers that are maximally tolerated

%%% paths
path_table = '/path/to/table/tbl.csv'; % path to table with all subject IDs, covariates, pacc5 score etc
path_mask = '/path/to/mask/mask.nii'; % path to mask with all relevant voxels (e.g. GM/task-active mask) for restricting the moderation analysis to those 
path_imgs = '/path/to/save/images'; % where the images of the additive and multiplicative (moderation) effect should be saved (principal component images are saved in the parent folder of that folder)
path_res = '/path/to/save/resultmat'; % path under which the results of the model and the principal components can be saved

path_func_template = '/path/to/con_images/%s.nii'; % path to contrast images; subject ID is inserted below

%% set up model variables
% import table with all cognitive and demographic information plus IDs
tbl = readtable(path_tbl,'TreatAsEmpty',{'NA'});
tbl.Properties.VariableNames{1} = 'ID'; % IDs are already in alphabetical order

% directly remove functional outliers from the table
if remove_outliers
    % create outlier list based on threshold specified above
    col = 'perc_outliers_extreme_dm';
    outlier_subjs = string(tbl.ID(tbl.(col) > thresh_outl));
    
    % get indices of outlier subjects
    if ~isempty(outlier_subjs)
        idc_outl = [];
        for ii=1:length(tbl.ID)
            if any(strcmp(tbl.ID{ii}, outlier_subjs))
                idc_outl = [idc_outl, ii];
            end
        end
        % remove outlier subjects; note: this excludes them from the table
        % that is saved later on
        tbl(idc_outl,:) = [];
    end
end


%%% Disease progression score
DP = tbl.DP.^2; % square

%%% Dependent variable: 
y = table2array(tbl(:,'pacc5'));

%% imaging modalities
% IDs were ordered alphabetically in both the table and in the folder
% anyways, I am making sure now that the functional measures correspond to
% the exact subjects in the table
n_subj = height(tbl);
files_func = cell(n_subj,1); 
for ii=1:n_subj
    files_func{ii} = sprintf(path_func_template, tbl.ID{ii});
end

x_func = spm_summarise(char(files_func), path_mask);

n_vox = size(x_func,2);

% for saving later
hdr = spm_data_hdr_read(path_mask);
mask = spm_data_read(path_mask);

%% covariates
if use_covs
    % convert site to categorical variable
    sites = tbl.site;
    sites_all = unique(sites);
    [~,idx_site] = max(sum(sites == sites_all'));
    sites_allbutone = setdiff(sites_all, sites_all(idx_site)); % 17 should be MD
    sites_ordinal = sites == sites_allbutone';
    % assemble covariates
    cov = [tbl.age_bl, tbl.TIV, tbl.sex, sites_ordinal]; % without education since it serves as a proxy for reserve
    cov_labels = ["age"; "TIV"; "sex"; ...
        strcat("site", strrep(string(num2str(sites_allbutone)), ' ', ''))];
else
    cov = [];
end

% center continuous covariates (i.e. only age and TIV here)
cov(:,1:2) = cov(:,1:2) - mean(cov(:,1:2));

%% model estimation
    
% separate into two groups: with only functional information or actually 
% usable for the model --> i.e. full information for all variables
pca_only = isnan(y) | isnan(DP) | any(isnan(cov),2);
idc_cmplt = find(~pca_only);

    
%%% use svd to reduce functional pattern
% center x_func
mean_func = mean(x_func);
x_func_orig = x_func; 
x_func = x_func - mean_func;
[U,S,V] = svd(x_func, 'econ');
% calculation of pcs for all participants
x_func_pc = U*S(:,1:n_pc); % principal components = XV = USV'V = US

%%% assemble independent variables + append covariates
X = [x_func_pc.*DP, x_func_pc, DP, ones(n_subj,1), cov];
X_labels = [strcat("dm_PC", strrep(string(num2str([1:n_pc]')), ' ', ''), "_by_DP"); ...
    strcat("dm_PC", strrep(string(num2str([1:n_pc]')), ' ', '')); ...
    "DP";
    "Intercept";
    cov_labels];
X = X(idc_cmplt,:);
y = y(idc_cmplt);

% determine coefficients
warning('off','all')
[b,se,tstat,pval] = lm_lean(X, y);
warning('on','all')

% save coefficients and PC weights
PCs = V(:,1:n_pc);
if ~exist(path_res), mkdir(path_res); end
save(fullfile(path_res, 'weights_pacc5.mat'), ...
    'PCs', 'mean_func', 'b', 'se', 'tstat', 'pval', 'X_labels')

%% analysis
% how much variance in the functional data do we capture with our 
% chosen principal components?
eig = diag(S).^2/(size(S,1)-1); % eigenvalues of covariance matrix: lambda_i = s_i^2/(N-1); division really does not matter for this part
fprintf('In total the first %d components explain %.2f%% of the variance in the DM contrast.\n', n_pc, sum(eig(1:n_pc))/sum(eig)*100)
for ii=1:n_pc
    fprintf('Component %d: %.2f%%\n', ii, eig(ii)/sum(eig)*100)
end

figure()
n = 25;
bar(1:n, eig(1:n)/sum(eig)*100)
xlabel('Principal component')
ylabel('Percent variance explained')
hold on
line(ones(1,2)*(n_pc+0.5), [0 100], 'Color', 'k', 'LineWidth', 1)
ylim([0 max(eig/sum(eig)*100)*1.25])
title('dm')
split = strsplit(path_imgs, filesep);
saveas(gcf, fullfile(split{1:end-1}, 'variance_by_PCs.png'))

y_pred = X*b;
ss_res = sum((y_pred-y).^2);
ss_tot = sum((y-mean(y)).^2);
fprintf('Total amount of variance in PACC5 explained by full model with %d components: %.2f.\n', n_pc, (1-ss_res/ss_tot)*100)

% map back the interaction term weights to voxel weights (as a combined sum)
% V is a matrix with column vectors = eigenvectors 
w_CR = V(:,1:n_pc)*b(1:n_pc); % weights for the multiplicative = CR effect
w_BAE = V(:,1:n_pc)*b(n_pc+1:2*n_pc); % weights for the additive effect

%% save images with principal component weights
% save the weights of the principal components in an image
vol = nan(size(mask));
if save_pc_imgs
    assert(n_vox==sum(mask(:)~=0))
    path_imgs_PC = fullfile(path_imgs, '..');
    for im=1:n_pc
       hdr.fname = fullfile(path_imgs_PC, ['PC' num2str(im) '.nii']);
       hdr.dt(1) = spm_type('float32');
       hdr.descrip = ['weights of principal component ' num2str(im) ' of the dm contrast'];
       vol(mask~=0) = V(:,im); % V = matrix of eigenvectors = weights of single voxels towards a single component im
       spm_data_write(hdr, vol);
    end
    
    % sum of weights for all selected components
    hdr.fname = fullfile(path_imgs_PC, ['PC_comb_' num2str(n_pc) '.nii']);
    hdr.dt(1) = spm_type('float32');
    hdr.descrip = ['sum of weights of principal components 1-' num2str(n_pc) ' of the dm contrast'];
    vol(mask~=0) = sum(V(:,1:n_pc),2); % V = matrix of eigenvectors = weights of single voxels towards a single component im
    spm_data_write(hdr, vol);
end

%% also save images with interaction coefficient mapped back to voxels
if ~exist(path_imgs), mkdir(path_imgs); end
hdr.fname = fullfile(path_imgs, ['vox_w_dm_by_DP_PC_comb_' num2str(n_pc) '.nii']);
hdr.dt(1) = spm_type('float32');
hdr.descrip = ['interaction terms mapped back to voxels based on PCs 1-' n_pc];
vol(mask~=0) = w_CR; 
spm_data_write(hdr, vol);

%% also save images with main effect coefficient (additive effect) mapped back to voxels
hdr.fname = fullfile(path_imgs, ['vox_w_dm_PC_comb_' num2str(n_pc) '.nii']);
hdr.dt(1) = spm_type('float32');
hdr.descrip = ['additive (main) effects mapped back to voxels based on PCs 1-' n_pc];
vol(mask~=0) = w_BAE; 
spm_data_write(hdr, vol);

%% create a CR score from the weights
% calculate the CR score (= multiplicative effect)
% also calculate the additive effect - let's call it BAE for brain activity
% effect

% weighted: weight by voxel weight of significant voxels
CR(:,1) = x_func * w_CR;
BAE(:,1) = x_func * w_BAE;
    
% save in table
colname = ['CR_pacc5_PC_comb_', num2str(n_pc), '_weighted'];
colname2 = ['BAE_pacc5_PC_comb_', num2str(n_pc), '_weighted'];
tbl.(colname) = CR(:,1);
tbl.(colname2) = BAE(:,1);

if save_tbl
    writetable(tbl, path_tbl)
end