% compute signal-to-noise ratio
% method: max over std of a region without signal

% input:
% beam_profile: one row in the 2D image that goes through the point target.
% usually a horizontal row

% output:
% snr_result: a single scalar value.

% review time: 03/28/2022

function snr_result = SNR(beam_profile)

% define signal as the maximum value of the beam profile
signal = max(beam_profile);

% take 1/8 of the beam profile as noise
noise = std(beam_profile(1:round(size(beam_profile,2)/8)));

snr_result = 20*log10(signal/noise); % Ashiqur suggested 20log10

end