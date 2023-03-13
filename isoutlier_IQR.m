function outlier = isoutlier_IQR(M, f_IQR)
    %%% determines if values in a matrix are outliers based on the
    %%% interquartile range
    %%% every column is treated as a separate variable
    
    %%% Inputs
    % M:        matrix of input variables that are to be checked for
    %           outliers
    % f_IQR:    multiple of interquartile range that should be used to
    %           declare a value an outlier; standard is 1.5, use 3 as a
    %           standard for extreme outliers
    
    %%% Output
    % outlier:  logical array indicating identified outliers with a 1


    % obtain quartiles
    quart_up = prctile(M, 75);
    quart_low = prctile(M, 25);
    IQR = quart_up - quart_low;
    lowerbound = quart_low - f_IQR*IQR;
    upperbound = quart_up + f_IQR*IQR;
    
    outlier = (M > upperbound | M < lowerbound);
end