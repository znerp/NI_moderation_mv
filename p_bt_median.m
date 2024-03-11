function p = p_bt_median(w,w_boot,rng_seed,do_plot)

% for each voxel: test weight from regression with original data against the
% sampled null distribution of weights for this voxel

% sort the bootstrap distributions for all voxels' weights by their median,
% and choose voxels whose median value lies in either the second or third 
% quartile
% then compare weights obtained from regression with the original data
% against this pseudo-null distribution by fitting a normal distribution to
% it and estimating the p value
% inspired by bootstrapping for multivariate mediation, but does not have
% to deal with non-identifiability of the coefficients, i.e. a normal
% distribution is fitted instead of a half-normal distribution

% see Chén Oliver Y., Crainiceanu Ciprian, Ogburn Elizabeth L., Caffo Brian S.,
% Wager Tor D., Lindquist Martin A. High-dimensional multivariate mediation
% with application to neuroimaging data. Biostatistics. 2018;19:121–136.


%%% Input parameters
% w         weights from model with original data
% w_boot    weights from models with bootstrapped data
% rng_seed  random seed (affects the indices of the randomly chosen voxels)
% do_plot   boolean to indicate if the pseudo-null distribution and the fitted
%           normal distribution should be plotted; default: False

% Output: 
% p      = p values estimated from comparison of a given weight to the
%          fitted normal distribution

if nargin < 3
   rng_seed = 176; 
end
if nargin < 4
   do_plot = 0;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% set random number generator to a certain number for reproducibility
rng(rng_seed)

n_vox = size(w,1);
% choose 10% of the voxels whose median value lies in either the second 
% or third quartile
[~,idxm] = sort(median(w_boot,2));
n_samp = round(n_vox/2/10);
imin = ceil(n_vox/4);
imax = floor(n_vox*3/4);
ivoxs = randi([imin,imax],n_samp,1);

% combine their distribution to create a pseudo-null distribution
null = w_boot(idxm(ivoxs),:);
null = null(:);

% fit a normal distribution (not half-normal bc we are not only looking  
% at the absolute values) to the pseudo-null and use this 
% distribution to estimate a p-value for each element of w
pd = fitdist(null,'Normal');
if do_plot
    % plot
    figure()
    histfit(null) % histfit uses fitdist to fit a distribution to data. Use fitdist to obtain parameters used in fitting.
end

% generate p values for the ws from the normal distribution
% two-sided test: b != mu
p_raw = normcdf(w,pd.mu,pd.sigma);
p = p_raw;
p(p > 0.5) = 1-p(p > 0.5);
