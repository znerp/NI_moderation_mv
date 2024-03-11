function [psig, lb, ub] = p_bt_voxelwise(w_boot, CI, plot)

% for each voxel: obtain confidence intervals for each voxel (separately)
% based on sorting the bootstrapped coefficients

%%% Input parameters
% w_boot    voxelwise coefficients from models with bootstrapped data
% CI        confidence intervals that should be determined; e.g. 0.95 or 95 for
%           95% confidence intervals
% plot      if true, the voxelwise coefficients of 10 random voxels are
%           plotted

% Output: 
% psig      inference of significance for each voxel (confidence intervals
%           should not contain 0)
% lb        lower bound of the confidence intervals
% ub        upper bound of the confidence intervals

if nargin < 3
    plot = 0;
end
if nargin < 2
    CI = 0.95;
end

if CI > 1
    CI = CI/100;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[n_vox, n_iter] = size(w_boot);

if n_iter < 1000
    warning('Using more bootstrap iterations than 999 is strongly recommended.')
end

if plot
    rng_vox = randi(n_vox, [10,1]);
end

% select interval that corresponds to 95% CI
lb_vox = floor((1-CI)/2 * n_iter) + 1; 
ub_vox = ceil((CI+(1-CI)/2) * n_iter);

% loop over voxels
psig = nan(n_vox,1); ub = psig; lb = ub;
for ivox=1:n_vox
    w_bt_sorted = sort(w_boot(ivox,:)); % default: ascending
    lb(ivox) = w_bt_sorted(lb_vox);
    ub(ivox) = w_bt_sorted(ub_vox);
    
    if plot
        if ismember(ivox, rng_vox)
            % plot
            figure()
            histfit(w_bt_sorted) 
        end
    end
end

% check if confidence intervals contain 0 
% <-- if they do, there is a change in sign btwn the lower and upper bound 
% <-- if their product is negative, there is a change in sign
% --> if product is positive, their confidence intervals don't contain 0
psig = lb.*ub > 0;

