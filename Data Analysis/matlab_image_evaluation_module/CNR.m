% compute contrast-to-noise ratio
% method: standard

% input:
% region_target: a 2D matrix of data from targeted region of interest.
% region_bkg: 2 2D matrix of data from background.

% output:
% cnr_result: a single scalar value.

% review time: 03/28/2022

function cnr_result = CNR(region_target,region_bkg)
    % compute mean of ROI
    mean_target = mean(region_target,'all');
    % compute mean of background
    mean_bkg = mean(region_bkg,'all');
    % compute standard deviation of ROI
    std_target = std(region_target,1,'all');
    % compute standard deviation of background
    std_bkg = std(region_bkg,1,'all');
    % compute CNR
    cnr_result = abs(mean_target-mean_bkg)/sqrt(std_target^2+std_bkg^2);
end


