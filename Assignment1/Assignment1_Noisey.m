function [] = Assignment1_Noisey()
%Assignment1 template
%   image_name       Full path/name of the input image (e.g. 'Test Image (1).JPG')


%% Load the input RGB image
image = imread('Test images/Test Image (1).png');

%% Add noise
variance = [0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0];
%% save a variable for the mean
mean_SSD = zeros(1, length(variance));
mean_NCC = zeros(1, length(variance));

    %% List all the template files starting with 'Template-' ending with '.png'
    % Assuming the images are located in the same directory as this m-file
    % Each template file name is accessible by templateFileNames(i).name
    templateFileNames = dir('Template images/Template-*.png');
    
    %% Get the number of templates (this should return 13)
    numTemplates = length(templateFileNames);

for count=1:size(variance, 2)
    image = imnoise(image, 'Gaussian', 0, variance(count));
    
    %% Create a gray-scale duplicate into grayImage variable for processing
    grayImage = rgb2gray(image);
    
    %% Set the values of SSD_THRESH and NCC_THRESH
    SSD_THRESH = 1508603;
    NCC_THRESH = 0.5;
    
%     %% Initialize two output images to the RGB input image
%     output_img1 = image;
%     output_img2 = image;
    
    %% Setup random number generation
%    rng(0, 'twister');
    
    %% For each template, do the following
    for i=1:numTemplates
        %% Load the RGB template image, into variable T
        filepath = sprintf('Template images/%s', templateFileNames(i).name);
        T = imread(filepath);
        
        %% Convert the template to gray-scale
        T = rgb2gray(T);
        
%         %% Extract the card name from its file name (look between '-' and '.' chars)
%         % use the cardName variable for generating output images
%         cardNameIdx1 = findstr(templateFileNames(i).name,'-') + 1;
%         cardNameIdx2 = findstr(templateFileNames(i).name,'.') - 1;
%         cardName = templateFileNames(i).name(cardNameIdx1:cardNameIdx2);
%         
%         %% Find the best match [row column] using Sum of Square Difference (SSD)
%         [SSDrow, SSDcol, ssd] = SSD(grayImage, T, SSD_THRESH);
%         
%         mean_SSD(count) = mean_SSD(count) + ssd;
        
        % If the best match exists
        % overlay the card name on the best match location on the SSD output image
        % Insert the card name on the output images (use small font size, e.g. 6)
        % set the overlay locations to the best match locations, plus-minus a random integer
        %position = [SSDcol+randi([-20, 20]), SSDrow+randi([-10, 10])];
        %output_img1 = insertText(output_img1, position, cardName);
        
        %% Find the best match [row column] using Normalized Cross Correlation (NCC)
        [NCCrow, NCCcol, ncc] = NCC(grayImage, T, NCC_THRESH);
        
        mean_NCC(count) = mean_SSD(count) + ncc;
        
        % If the best match exists
        % overlay the card name on the best match location on the NCC output image
        % Insert the card name on the output images (use small font size, e.g. 6)
        % set the overlay locations to the best match locations, plus-minus a random integer
%         position = [NCCcol+randi([-20, 20]), NCCrow+randi([-20, 20])];
%         output_img2 = insertText(output_img2, position, cardName);
        
        
    end
end

mean_SSD = mean_SSD / numTemplates;
%     mean_NCC = mean_NCC / numTemplates;

disp(mean_SSD);
disp(variance);

%% Display the plot
scatter(variance, mean_SSD);
%figure
%scatter(variance, mean_NCC);

%% Display the output images
%     imshow(output_img1);
%     figure
%     imshow(output_img2);
end

%% Implement the SSD-based template matching here
function [SSDrow, SSDcol, SSD] = SSD(grayImage, T, SSD_THRESH)
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

for row = half_Tr:(Gray_row-half_Tr-1)
    for col = half_Tc:(Gray_col-half_Tc-1)
        patch = grayImage(row-half_Tr+1:row+half_Tr+1, col-half_Tc+1:col+half_Tc+1);
        squared_diff = (T - patch).^2;
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

SSD = min_ssd;
end

%% Implement the NCC-based template matching here
function [NCCrow, NCCcol, NCC] = NCC(grayImage, T, NCC_THRESH)
% inputs
%           grayImag        gray-scale image
%           T               gray-scale template
%           NCC_THRESH      threshold above which a match is accepted
% outputs
%           NCCrow          row of the best match (empty if unavailable)
%           NCCcol          column of the best match (empty if unavailable)

T_row = size(T,1);
T_col = size(T,2);
half_Tr = idivide(T_row, int32(2));
half_Tc = idivide(T_col, int32(2));
Gray_row = size(grayImage,1);
Gray_col = size(grayImage,2);
max_ncc = realmin;
max_row = 0;
max_col = 0;

T_avg = zeros(T_row, T_col);
T_avg(:) = mean(mean(T));
T_diff = double(T) - T_avg;
T_square = sum(sum(T_diff .^ 2));

for row = half_Tr:(Gray_row-half_Tr-1)
    for col = half_Tc:(Gray_col-half_Tc-1)
        patch = grayImage(row-half_Tr+1:row+half_Tr+1, col-half_Tc+1:col+half_Tc+1);
        
        im_avg = zeros(T_row, T_col);
        im_avg(:) = mean(mean(patch));
        im_diff = double(patch) - im_avg;
        im_square = sum(sum(im_diff.^2));
        
        numerator = sum(sum((T_diff).*(im_diff)));
        denominator = sqrt(T_square * im_square);
        
        ncc = numerator / denominator;
        
        if ncc > max_ncc
            max_ncc = ncc;
            max_row = row;
            max_col = col;
        end
    end
end

NCCrow = -1;
NCCcol = -1;
if (max_ncc >= NCC_THRESH)
    disp(max_ncc);
    
    NCCrow = max_row;
    NCCcol = max_col;
end

NCC = max_ncc;
end
