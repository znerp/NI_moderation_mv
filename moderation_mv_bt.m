% Optional 4th step of the multivariate moderation model framework
%%% Bootstrapping on the voxels' moderation (CR) coefficients %%%

% The idea is to reduce the fMRI contrast in the chosen voxels to a number 
% of principal components and use them in a multivariate regression. The 
% regression weights are then be mapped back onto the voxels to determine 
% their contribution. 
% These moderation coefficients of the distinct voxels undergo a
% bootstrapping approach in this script in order to allow inference about
% their significance. The bootstrapping procedure is adapted from the
% bootstrapping approach in the context of multivariate mediation presented
% by Chén et al. (2018):
% Chén Oliver Y., Crainiceanu Ciprian, Ogburn Elizabeth L., Caffo Brian S.,
% Wager Tor D., Lindquist Martin A. High-dimensional multivariate mediation
% with application to neuroimaging data. Biostatistics. 2018;19:121–136.


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

use_covs = 1; % should covariates be used?
n_pc = 4; % number of principal components (4 or 2 performed best)
n_iter = 5000; % iterations for bootstrapping
save_pvals = 0;

% outliers are no longer identified manually, but excluded based on
% information saved in the table and a threshold indicated below
remove_outliers = 1
thresh_outl = 10; % percent of voxels with extreme outliers that are maximally tolerated

%%% paths
path_tbl = '../../data/m0/tables/multivariate/covs_arcsin_CSF_s6_sub493_dm.csv'; % table with covariates, pacc5 score etc
path_mask = '../../data/masks/mask_GM35_shoot_0.2_noNaNdm.nii'; % mask to restrict voxel-wise comparison (NaNs in any of the functional images were additionally masked out
path_imgs = '../../images/10_CRnetwork_paramGLM/multivariate/dm/moderation_pacc5'; % where the images should be saved
path_res = '../../data/results/CRnetwork_mv/dm'; % path where the principal components and the results of the model can be saved

path_struct_template = '../../data/m0/derivatives/prepr/%s/anat/r3p5rc1anat.nii'; % insert subject ID with sprintf below
path_func_template = '../../data/m0/derivatives/glm/contrasts/arcsin_CSF_s6/s6_dm_%s.nii'; % insert subject ID with sprintf below

rng(617)
%% set up model variables
% import table with delta_pacc5
tbl = readtable(path_tbl,'TreatAsEmpty',{'NA'});
tbl.Properties.VariableNames{1} = 'ID'; % IDs are already in alphabetical order

% directly remove functional outliers from the table
if remove_outliers
    % create outlier list based on threshold specified above
    col = 'perc_outliers_extreme_dm';
    outlier_subjs = string(tbl.ID(tbl.(col) > thresh_outl));
    
    % get indices of outlier subjects
    if length(outlier_subjs) > 0
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


%%% pathological load score
DP = tbl.DP.^2; % square

%%% Dependent variable: 
y = table2array(tbl(:,'pacc5'));

%% imaging modalities
% IDs were ordered alphabetically in both the table and in the folder
% anyways, I am making sure now that the functional measures correspond to
% the exact subjects in the table
n_subj = height(tbl);
files_func = cell(n_subj,1); files_struct = cell(n_subj,1);
for ii=1:n_subj
    files_func{ii} = sprintf(path_func_template, tbl.ID{ii});
    files_struct{ii} = sprintf(path_struct_template, tbl.ID{ii});
end

x_struct = spm_summarise(char(files_struct), path_mask);
x_func = spm_summarise(char(files_func), path_mask);

n_vox = size(x_func,2);
% assert(all(size(x_struct) == size(x_func)))

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

%% SVD
% separate into two groups: with only functional information or actually
% usable for the model --> i.e. full information for all variables
pca_only = isnan(y) | isnan(DP) | any(isnan(cov),2);
idc_cmplt = find(~pca_only);


%%% use svd to reduce functional pattern
% center x_func
mean_func = mean(x_func);
x_func = x_func - mean_func;
[U,S,V] = svd(x_func, 'econ');
% calculation of pcs for all participants
x_func_pc = U*S(:,1:n_pc); % principal components = XV = USV'V = US

%%% assemble independent variables + append covariates
X = [x_func_pc.*DP, x_func_pc, DP, ones(n_subj,1), cov];
X_labels = [strcat(string('dm_PC'), strrep(string(num2str([1:n_pc]')), ' ', ''), "_by_DP"); ...
    strcat(string('dm_PC'), strrep(string(num2str([1:n_pc]')), ' ', '')); ...
    "DP";
    "Intercept";
    cov_labels];

X = X(idc_cmplt,:);
y = y(idc_cmplt);

%% run real model
% determine coefficients
warning('off','all')
[b,se,tstat,pval] = lm_lean(X, y);
warning('on','all')

w = V(:,1:n_pc)*b(1:n_pc);
%% bootstrapping
[N,P] = size(x_func);
w_boot = nan(P,n_iter);
f = waitbar(0,'Peforming bootstrap iterations.');
for it=1:n_iter
    % 1. sample subjects with replacement from the low dimensional space
    [Xboot,idx] =  datasample(X,N); % samples WITH replacement by default
    yboot = y(idx);
    % 2. estimate coefficients with bootstrap sample
    warning('off','all')
    [b,~,~,~] = lm_lean(Xboot, yboot);
    warning('on','all')
    % 3. map back to the original voxel space and stack horizontally in
    % matrix; only for the components whose interaction effects was
    % significant in the unshuffled sample
    w_boot(:,it) = V(:,1:n_pc)*b(1:n_pc);
    
    % progress
    waitbar(it/n_iter,f)
end
close(f)

%% p values from bootstrap
% 4. for each voxel: test weight from regression with real data against the
% samples null distribution of weights for this voxel
% unlike in the paper, we should not have the problem that the sign of the
% coefficient is unidentifiable (this is just due to the mediation
% framework in the paper)

% Next, we sort the bootstrap distributions for all voxels by their median,
% and choose voxels whose median value lies in either the second or third 
% quartile.
% sort bootstrap distribution for all voxels by their median
p_all = p_bt_median(w,w_boot,1); % third argument for plotting
p_sig = p_all<0.05;
fprintf('\n%d voxels contribute significantly to CR.\n', sum(p_sig))

%% save images
vol = nan(size(mask));
% 1. image with p values
hdr.fname = fullfile(path_imgs, ['p_dm_by_DP_PC_comb' num2str(n_pc) '.nii']);
hdr.dt(1) = spm_type('float32');
hdr.descrip = ['p values for dm_by_DP from bootstrapping'];
vol(mask~=0) = p_all;
spm_data_write(hdr, vol);
% 2. binary image indicating which p values were significant
hdr.fname = fullfile(path_imgs, ['psig05_dm_by_DP_PC_comb' num2str(n_pc) '.nii']);
hdr.dt(1) = spm_type('uint8');
hdr.descrip = ['mask of p values < 0.05 for dm_by_DP from bootstrapping'];
vol(mask~=0) = p_sig;
spm_data_write(hdr, vol);
   

