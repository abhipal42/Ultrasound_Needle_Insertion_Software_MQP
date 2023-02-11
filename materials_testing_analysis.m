
clear all
close all

%adding path
addpath("Data Analysis\matlab_image_evaluation_module\")

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Insert image load
image = imread("Data\materials_test_V2\ground_truth_materialTest.jpeg");

fig_target = image;
hold on

imagesc(image)
axis ij
% Show image (e.g. imshow)
[x,y] = ginput; % Call ginput for interactive registration (The position you click is stored as x1 and y1)

x1 = x(1);
y1 = y(1);

x2 = x(2);
y2 = y(2);

%coordinates for horizontal line
x3 = x(3);
y3 = y(3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
half_width = 6;
half_height = 6;

% In my case, I define a rectangle window around the selected pointdet_left = round(x1) - half_width;
det_right = round(x1) + half_width;
det_left = round(x1) - half_width;
det_top = round(y1) - round(half_height);
det_bottom = round(y1) + round(half_height);

det_right2 = round(x2) + half_width;
det_left2 = round(x2) - half_width;
det_top2 = round(y2) - round(half_height);
det_bottom2 = round(y2) + round(half_height);

% Draw rectangle
rectangle('Position',[det_left det_top half_width*2 half_height*2],'EdgeColor','r');
rectangle('Position', [det_left2 det_top2 half_width*2 half_height*2],'EdgeColor','g');

%draw line
line = yline(y3, 'Color', 'yellow');
lineVector = fig_target(round(y3),500:800);


% Save figure for reference (saving the defined window image would be helpful as well)
hold off

target_ROI = fig_target(det_top:det_bottom, det_left:det_right);
background_ROI = fig_target(det_top2:det_bottom2, det_left2:det_right2);

%convert uint8 to double
target_ROI = double(target_ROI);
background_ROI = double(background_ROI);
lineVector = double(lineVector);

figure
imagesc(target_ROI);
% colormap gray

figure 
imagesc(background_ROI)
% colormap gray

figure
plot(lineVector)

%%%%% CNR CALCULATION %%%%%%%%
cnr_value = CNR(target_ROI, background_ROI)
gt_cnr = 5.4607;

relative_percent = (cnr_value/gt_cnr) * 100


%%%%% FWHM and SNR CALCULATIONS %%%%%%%
fwhm_value = FWHM(lineVector) %unit is in pixels --> later can convert pixel to mm

relative_percent = fwhm_value

% fwhm_value2 = FWHM_gaussian_fitting(lineVector)

snr_value = SNR(lineVector)

rmpath("Data Analysis\matlab_image_evaluation_module\")
