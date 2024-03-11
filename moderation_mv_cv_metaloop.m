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

% outliers are excluded based on information saved in the table (requires 
% outliers_beta to be run first) and a threshold indicated below
remove_outliers = 1
thresh_outl = 10; % percent of voxels with extreme outliers that are maximally tolerated

%%% paths
path_table = '/path/to/table/tbl.csv'; % path to table with all subject IDs, covariates, pacc5 score etc
path_mask = '/path/to/mask/mask.nii'; % path to mask with all relevant voxels (e.g. GM/task-active mask) for restricting the moderation analysis to those 

path_func_template = '/path/to/con_images/%s.nii'; % path to contrast images; subject ID is inserted below

savepath_plots = '/path/for/saving/plots'; % where should the plots for the cross-validation metrics be saved?
% set seed (for cv-partition)
rng(617)

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

%% model estimation in cross-validation scheme
    
% separate into two groups: with only functional information or actually 
% usable for the model --> i.e. full information for all variables
% (important in order to have similar proportions of those in all folds)
pca_only = isnan(y) | isnan(DP) | any(isnan(cov),2);
idc_cmplt = find(~pca_only);

% initialize variables for evaluating prediction accuracy
cv_pred = nan(n_iter,n_subj,max_pc); cv_pred_noInt = cv_pred;
cv_Rsq = nan(n_iter,K,max_pc); cv_R_noInt = cv_Rsq;
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
            warning('on','all')
            % predict
            y_pred = X_te*b;
            
            
            % collect results
            cv_pred(it,idc_te_cmplt,n_pc) = y_pred;
            
            progress = (ifold-1)*max_pc + n_pc/max_pc;
            goal = max_pc*K;
            waitbar(progress/goal,f)
        end
        
    end
    close(f)
end


%% comment reviewer 1
% "one should look at performance (loss) measured on 
% all out-of-sample data in aggregate. This allows the performance to be 
% estimated across the same amount of data that is observed, while still 
% being all left-out data"

y_cpmlt = y(idc_cmplt);
ss_tot = sum((y_cpmlt-mean(y_cpmlt)).^2);
cv_Rsq = nan(n_iter,n_pc);
cv_mae = cv_Rsq;
cv_rmse = cv_Rsq;
for it=1:n_iter
    pred = reshape(cv_pred(it,idc_cmplt,:), [length(idc_cmplt), n_pc]);
    
    ss_res = sum((pred - y_cpmlt).^2);
    cv_Rsq(it,:) = 1 - ss_res/ss_tot;
    
    cv_mae(it,:) = mean(abs(pred-y_cpmlt));
    
    cv_rmse(it,:) = sqrt(mean((pred-y_cpmlt).^2));
    
end

%% plots for oefficient of determination and RMSE
g = reshape(repmat(1:max_pc,[n_iter 1]),[numel(cv_Rsq) 1]);

% R^2 in dependence of number of components
figure('Position', [0 0 120*max_pc/5 480])
boxplot(reshape(cv_Rsq,[numel(cv_Rsq) 1]), g)
hold on
plot(1:max_pc, mean(cv_Rsq), 'k-o') % plot means as well
xlabel('Number of PCs used in linear model')
ylabel('Crossvalidation R^2')
if ~exist(savepath_plots), mkdir(savepath_plots); end
saveas(gcf, fullfile(savepath_plots, 'CV_Rsq.png'))

% RMSE in dependence of number of components
figure('Position', [0 0 120*max_pc/5 480])
boxplot(reshape(cv_rmse,[numel(cv_rmse) 1]), g)
hold on
plot(1:max_pc, mean(cv_rmse), 'k-o') % plot means as well
xlabel('Number of PCs used in linear model')
ylabel('RMSE in prediction of pacc5')
saveas(gcf, fullfile(savepath_plots, 'CV_RMSE.png'))

mean_cv_Rsq = mean(cv_Rsq);
[~,idx] = sort(mean_cv_Rsq, 'descend');

for ii=1:5
    fprintf('%d. Number of PCs: %d \tR^2: %.4f\n', ii, idx(ii), mean_cv_Rsq(idx(ii)))
end
