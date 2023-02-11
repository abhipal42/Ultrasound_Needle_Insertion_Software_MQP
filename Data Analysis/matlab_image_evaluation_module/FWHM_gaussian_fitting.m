% compute full-width-at-half-maximum
% evaluate resolution of imaging system
% method: fitting a guassian model to the beam profile data
% purpose: to handle the case where the peak range is narrow
% and does not have sufficient samples to accurately calculate FWHM.

% input: 
% beamprofile: one row in the 2D image that goes through the point target.
% usually a horizontal row

% output: a single scalar value

% review time: 07/18/2022

function output = FWHM_gaussian_fitting(beam_profile)
    
    f = fit([1:size(beam_profile,2)]',beam_profile','gauss1');
    output = 2*sqrt(log(2))*f.c1;
    
end

