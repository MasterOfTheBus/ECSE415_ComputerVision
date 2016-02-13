function [] = NormSSD(image_name)
%Assignment1 template
%   image_name       Full path/name of the input image (e.g. 'Test Image (1).JPG')


%% Load the input RGB image
image = imread(image_name);

%% Create a gray-scale duplicate into grayImage variable for processing
grayImage = rgb2gray(image);
grayImage = int32(grayImage);

%% List all the template files starting with 'Template-' ending with '.png'
% Assuming the images are located in the same directory as this m-file
% Each template file name is accessible by templateFileNames(i).name
templateFileNames = dir('Template images/Template-*.png');

%% Get the number of templates (this should return 13)
numTemplates = length(templateFileNames);

%% Set the values of SSD_THRESH and NCC_THRESH
SSD_THRESH = 15000000;

%% Initialize two output images to the RGB input image
output_img1 = image;

%% Setup random number generation
rng(0, 'twister');

%% For each template, do the following
for i=1:numTemplates
    %% Load the RGB template image, into variable T
    filepath = sprintf('Template images/%s', templateFileNames(i).name);
    T = imread(filepath);
    
    %% Convert the template to gray-scale
    T = rgb2gray(T);
    T = int32(T);
    
    %% Extract the card name from its file name (look between '-' and '.' chars)
    % use the cardName variable for generating output images
    cardNameIdx1 = findstr(templateFileNames(i).name,'-') + 1;
    cardNameIdx2 = findstr(templateFileNames(i).name,'.') - 1;
    cardName = templateFileNames(i).name(cardNameIdx1:cardNameIdx2); 
    
    %% Find the best match [row column] using Sum of Square Difference (SSD)
    [SSDrow, SSDcol] = SSD(grayImage, T, SSD_THRESH);
    
    % If the best match exists
    % overlay the card name on the best match location on the SSD output image                      
    % Insert the card name on the output images (use small font size, e.g. 6)
    % set the overlay locations to the best match locations, plus-minus a random integer 
    position = [SSDcol+randi([-30, 30]), SSDrow+randi([-30, 30])];
    output_img1 = insertText(output_img1, position, cardName);    
end

%% Display the output images 
imshow(output_img1);

end

%% Implement the SSD-based template matching here
function [SSDrow, SSDcol] = SSD(grayImage, T, SSD_THRESH)
% inputs
%           grayImag        gray-scale image
%           T               gray-scale template
%           SSD_THRESH      threshold below which a match is accepted
% outputs
%           SSDrow          row of the best match (empty if unavailable)
%           SSDcol          column of the best match (empty if unavailable)

% first step is to compute the SSD matrix
T_row = size(T,1);
T_col = size(T,2);
half_Tr = idivide(T_row, int32(2));
half_Tc = idivide(T_col, int32(2));
Gray_row = size(grayImage,1);
Gray_col = size(grayImage,2);
min_ssd = realmax;
min_row = 0;
min_col = 0;

T_mean = zeros(T_row, T_col);
T_mean(:) = mean(mean(T));
T_diff = double(T) - T_mean;
patch_mean = zeros(T_row, T_col);

for row = half_Tr:(Gray_row-half_Tr-1)
    for col = half_Tc:(Gray_col-half_Tc-1)
        patch = grayImage(row-half_Tr+1:row+half_Tr+1, col-half_Tc+1:col+half_Tc+1);
        patch_mean(:) = mean(mean(patch));
        
        squared_diff = (T_diff - double(patch) + patch_mean).^2;
        ssd = sum(sum(squared_diff));
        if ssd < min_ssd
           min_ssd = ssd;
           min_row = row;
           min_col = col;
        end
    end
end

    SSDrow = -1;
    SSDcol = -1;
    if (min_ssd <= SSD_THRESH)
       disp(min_ssd);
       
       SSDrow = min_row;
       SSDcol = min_col;
    end
end
