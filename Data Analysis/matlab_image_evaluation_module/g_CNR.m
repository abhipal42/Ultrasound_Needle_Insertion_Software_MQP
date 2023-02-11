% generalized contrast-to-noise ratio
% method: compare histogram overlap between two data clusters
% advantage: can be used to compare image beamformed with DAS (linear), DAS+CF (unlinear), and
% DMAS (unlinear) or other methods.
% paper: "The Generalized Contrast-to-Noise Ratio", AR-M,2018 IUS

% compare ROI and background image patch (2D data)
% review time: 03/28/2022
function GCNR = g_CNR(region_target,region_bkg)
    
    % find lower and upper limit of two distributions
    lower = min(min(region_target(:)),min(region_bkg(:)));
    upper = max(max(region_target(:)),max(region_bkg(:)));
    lower = round(lower);
    upper = round(upper);
    
    % number of bins should be an integer
    %num_bins = upper - lower + 1;
    dx = 1;
    histo_edge = [lower:dx:upper];
    
    % generate histogram
    h_target = histogram(region_target,histo_edge,'Normalization','pdf');
    h_target_value = h_target.Values;
    
    h_bkg = histogram(region_bkg,histo_edge,'Normalization','pdf');
    h_bkg_value = h_bkg.Values;
    
    % compute OVL
    OVL = 0;
    for i = 1:size(histo_edge,2)-1
        value = min(h_target_value(i),h_bkg_value(i))*dx;
        OVL = OVL + value;
    end
    
    % output gcnr
    GCNR = 1 - OVL;
    
end