% 2nd step of the multivariate moderation model framework:
%%% Determination of the optimal number of principal components %%%

% The idea here is to reduce the fMRI contrast in the chosen voxels
% to a number of principal components and use them in a multivariate regression. 
% The optimal number of principal components is sought to be determined via
% 10-fold cross-validation in this script. 
% The cross-validation is repeated 10 times to increase statistical
% robustness.
% SVD is always performed on the
% training set and then metrics like R^2, MAE and RMSE are calculated for
% every combination of fold and number of PCs, but also across all folds
% (because every data point is predicted exactly once in this procedure)
% for each number of PCs. 


%%% multivariate moderation model:
% cognition = b0 + b1*DP + b2_1*fmri_PC1 + b3_1*DP*fmri_PC1 + b2_2*fmri_PC2 + 
% b3_2*DP*fmri_PC2 + ... + b2_n*fmri_PC1 + b3_n*DP*fmri_PC1 + C*cov

% DV: cognition, e.g. pacc5
% IVs:  DP score (later named PL), 
%       principal component of DM contrast,
%       interaction between DP dm_PC
% covs: age
%       sex
%       TIV
%       sites



%% initialize variables
clear; close all 
 
use_covs = 1; % should covariates be used?
max_pc = 25; % maximal number of principal components tested
K = 10; % number of folds for k-fold cross validation
n_iter = 10; % number of iterations for the metaloop

% outliers are no longer identified manually, but excluded based on
% information saved in the table and a threshold indicated below
remove_outliers = 1
thresh_outl = 10; % percent of voxels with extreme outliers that are maximally tolerated

%%% paths
path_tbl = '../../data/m0/tables/multivariate/covs_arcsin_CSF_s6_sub493_dm.csv'; % table with covariates, pacc5 score etc
path_mask = '../../data/masks/mask_GM35_shoot_0.2_noNaNdm.nii'; % mask to restrict voxel-wise comparison (NaNs in any of the functional images were additionally masked out

path_struct_template = '../../data/m0/derivatives/prepr/%s/anat/r3p5rc1anat.nii'; % insert subject ID with sprintf below
path_func_template = '../../data/m0/derivatives/glm/contrasts/arcsin_CSF_s6/s6_dm_%s.nii'; % insert subject ID with sprintf below

savepath_plots = '../../images/10_CRnetwork_paramGLM/multivariate/dm/moderation_pacc5'; % where should the plots for the cross-validation metrics be saved?
% set seed (for cv-partition)
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
        % remove outlier subjects
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

%% model estimation in cross-validation scheme
    
% separate into two groups: with only functional information or actually 
% usable for the model --> i.e. full information for all variables
% (important in order to have similar proportions of those in all folds)
pca_only = isnan(y) | isnan(DP) | any(isnan(cov),2);
idc_cmplt = find(~pca_only);

% initialize variables for evaluating prediction accuracy
cv_pred = nan(n_iter,n_subj,max_pc); cv_pred_noInt = cv_pred;
cv_R = nan(n_iter,K,max_pc); cv_R_noInt = cv_R;
cv_mae = nan(n_iter,K,max_pc); cv_mae_noInt = cv_mae;
cv_rmse = nan(n_iter,K,max_pc); cv_rmse_noInt = cv_rmse;
% implement a metaloop that splits the data into folds differently at
% different stages
for it=1:n_iter

    f = waitbar(0,['Iteration ' num2str(it) ' running.']);
    % cross validation
    cv_p = cvpartition(pca_only,'KFold',K); % creates K folds with roughly the same proportions of the different groups in all folds
    for ifold=1:cv_p.NumTestSets
        
%         fprintf('\nCalculating fold %d.\n', ifold);
        
        idc_tr  = cv_p.training(ifold);
        idc_te  = cv_p.test(ifold);
        
        % find valid indices for the model (i.e. only complete observations)
        idc_tr_cmplt = intersect(idc_cmplt, find(idc_tr));
        idc_te_cmplt = intersect(idc_cmplt, find(idc_te));
        
        %%% use svd to reduce functional pattern; perform svd only on the
        %%% (whole) training set
        % center x_func
        x_func_tr = x_func(idc_tr,:);
        x_func_tr = x_func_tr - mean(x_func_tr);
        [U,S,V] = svd(x_func_tr, 'econ');
        
        for n_pc=1:max_pc
            % calculation of pcs for all participants
            x_func_pc_all = x_func*V(:,1:n_pc); % principal components = XV = USV'V = US
            
            %%% assemble independent variables + append covariates
            X = [x_func_pc_all.*DP, x_func_pc_all, DP, ones(n_subj,1), cov];
            X_labels = [strcat(string(['dm_PC']), strrep(string(num2str([1:n_pc]')), ' ', ''), "_by_DP"); ...
                strcat(string(['dm_PC']), strrep(string(num2str([1:n_pc]')), ' ', '')); ...
                "DP";
                "Intercept";
                cov_labels];
            
            X_tr = X(idc_tr_cmplt,:);
            X_te  = X(idc_te_cmplt,:);
            y_tr = y(idc_tr_cmplt,:);
            y_te  = y(idc_te_cmplt,:);
            
            % determine coefficients
            % two models: one with, one without interactions (in order to
            % compare their performance afterwards)
            warning('off','all')
            [b,~,~,~] = lm_lean(X_tr, y_tr);
            [b_noInt,~,~,~] = lm_lean(X_tr(:,n_pc+1:end), y_tr);
            warning('on','all')
            % predict
            y_pred = X_te*b;
            y_pred_noInt = X_te(:,n_pc+1:end)*b_noInt;
            
            
            % collect results
            cv_pred(it,idc_te_cmplt,n_pc) = y_pred;
            ss_res = sum((y_te-y_pred).^2);
            ss_tot = sum((y_te-mean(y_te)).^2);
            cv_R(it,ifold,n_pc) = 1 - ss_res/ss_tot;
            cv_mae(it,ifold,n_pc) = mean(abs(y_pred-y_te));
            cv_rmse(it,ifold,n_pc) = sqrt(mean((y_pred-y_te).^2));
            
            cv_pred_noInt(it,idc_te_cmplt,n_pc) = y_pred_noInt;
            cv_R_noInt(it,ifold,n_pc) = corr(y_pred_noInt, y_te)^2;
            cv_mae_noInt(it,ifold,n_pc) = mean(abs(y_pred_noInt-y_te));
            cv_rmse_noInt(it,ifold,n_pc) = sqrt(mean((y_pred_noInt-y_te).^2));
            
            %          fprintf('Number of principal components: %d\n', n_pc)
            progress = (ifold-1)*max_pc + n_pc/max_pc;
            goal = max_pc*K;
            waitbar(progress/goal,f)
        end
        
    end
    close(f)
end

%% plot results 
% reshape matrices
R_sq_all = reshape(cv_R, [n_iter*K n_pc]);
mae_all = reshape(cv_mae, [n_iter*K n_pc]);
rmse_all = reshape(cv_rmse, [n_iter*K n_pc]);

g = reshape(repmat(1:max_pc,[n_iter*K 1]),[numel(R_sq_all) 1]);
% R squared in dependence of number of components
figure('Position', [0 0 120*max_pc/5 480])
boxplot(reshape(cv_R*100,[numel(cv_R) 1]), g)
hold on
plot(1:max_pc, mean(R_sq_all)*100, 'k-o') % plot means as well
xlabel('Number of PCs used in linear model')
ylabel(['Percent variance in pacc5 explained by model in test set'])
saveas(gcf, fullfile(savepath_plots, 'CV_Rsq.png'))

% RMSE in dependence of number of components
figure('Position', [200 200 120*max_pc/5 480])
boxplot(reshape(cv_rmse,[numel(cv_rmse) 1]), g)
hold on
plot(1:max_pc, mean(rmse_all), 'k-o') % plot means as well
xlabel('Number of PCs used in linear model')
ylabel('RMSE in prediction of pacc5')
saveas(gcf, fullfile(savepath_plots, 'CV_RMSE.png'))


%% little statement about which model is best
mean_R_sq = reshape(mean(mean(cv_R,1)), [1 n_pc]);
mean_rmse = reshape(mean(mean(cv_rmse,1)), [1 n_pc]);
[~,idx] = sort(mean_R_sq, 'descend');
[~,idx2] = sort(mean_rmse, 'ascend');

mean_R_sq(idx(1:3))
idx(1:3)
mean_rmse(idx2(1:3))
idx2(1:3)


[m1,i1] = max(R_sq_all);
[m2,i2] = min(rmse_all);

% fprintf('The best model in terms of explained variance in PACC5 for the whole dataset is one with %d PCs.\n It reaches an R^2 of %.2f.\n\n', i1, m1)
% fprintf('The best model in terms of RMSE in PACC5 predictions for the whole dataset is one with %d PCs.\n It reaches an RMSE of %.2f.\n\n', i2, m2)
% fprintf('The best model in terms of differences in explained variance in PACC5 for the whole dataset with the full model vs the additive model is one with %d PCs.\n The R^2 of the full model is %.2f higher.\n\n', i3, m3)
% fprintf('The best model in terms of differences in RMSE in PACC5 predictions for the whole dataset with the full model vs the additive model is one with %d PCs.\n The RMSE of the full model is %.2f lower.\n\n', i4, m4)
% fprintf('Number of PCs: %d \nR^2: %.4f\n', i1, m1)
% fprintf('Number of PCs: %d \nRMSE: %.4f\n', i2, m2)
% fprintf('Number of PCs: %d \nR^2 difference: %.4f\n', i3, m3)
% fprintf('Number of PCs: %d \nRMSE difference: %.4f\n', i4, m4)
