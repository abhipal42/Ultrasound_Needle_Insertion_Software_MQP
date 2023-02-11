% compute full-width-at-half-maximum
% evaluate resolution of imaging system
% method: find the first 50% location from both sides of the signal

% input: 
% beamprofile: one row in the 2D image that goes through the point target.
% usually a horizontal row

% output: a single scalar value

% review time: 03/28/2022
function output = FWHM(beam_profile)

% apply interpolation for finer resolution
ratio = 1;
beam_profile = imresize(beam_profile,[size(beam_profile,1),ratio*size(beam_profile,2)]);

% allocate data space
fst_crs = 0;
sec_crs = 0;

% find the half maximum value
maxval = max(beam_profile);
lev50 = maxval*0.5;

% find the half maximum from start
j = 1;
while beam_profile(j)-lev50 < 0
    j = j+1;
end
fst_crs = j;
    
% flip the beam profile
beam_profile = flip(beam_profile,2);

j = 1;
while beam_profile(j)-lev50 < 0
    j = j+1;
end
sec_crs = j;
sec_crs = size(beam_profile,2) - sec_crs;

output = (sec_crs - fst_crs)./ratio;

end

