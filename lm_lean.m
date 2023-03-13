function [b, se, t_stat, p_val] = lm_lean(X,Y)
    %%% Fit a linear model and calculate standard error, t statistic as
    %%% well as p values; faster than fitlm
    %%% intercept has to be manually included in X
    %%% formula assembled from Learning Statistics with R, page 468
    
    %%% Inputs
    % X:        design matrix of inputs
    % Y:        outputs
    
    %%% Outputs
    % b:        coefficients of the lm
    % se:       standard error
    % t_stat:   t statistic
    % p_val:    p value

    b = X\Y; % same as b = pinv(X)*Y;

    e = Y - X*b; % residuals
    N = size(X,1); % number of observations
    K = size(X,2); % number of predictors (with intercept)
    res_variance = e'*e / (N-K); % residual variance; K = predictors + INTERCEPT
    coeff_covariance = inv(X'*X); % take real inverse, not pseudo-inverse!
    cov_mat_coeff = res_variance * coeff_covariance; 
    se = sqrt(diag(cov_mat_coeff)); 
    t_stat = b ./ se;
    p_val = (1 - tcdf(abs(t_stat),N-K))*2; % two-sided t test; H1: b!=0 (not b>0 or b<0)

end