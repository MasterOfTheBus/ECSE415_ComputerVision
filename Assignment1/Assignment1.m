function [] = Assignment1(image_name)
%Assignment1 template
%   image_name       Full path/name of the input image (e.g. 'Test Image (1).JPG')


%% Load the input RGB image
image = imread(image_name);

%% Create a gray-scale duplicate into grayImage variable for processing
grayImage = rgb2gray(image);

%% List all the template files starting with 'Template-' ending with '.png'
% Assuming the images are located in the same directory as this m-file
% Each template file name is accessible by templateFileNames(i).name
templateFileNames = dir('Template images/Template-*.png');

%% Get the number of templates (this should return 13)
numTemplates = length(templateFileNames);

%% Set the values of SSD_THRESH and NCC_THRESH
SSD_THRESH = 10;
NCC_THRESH = 10;

%% Initialize two output images to the RGB input image
output_img1 = image;
output_img2 = image;

%% For each template, do the following
for i=1:numTemplates
    %% Load the RGB template image, into variable T
    path = sprintf('Template images/%s', templateFileNames(i).name);
    T = imread(path);
    
    %% Convert the template to gray-scale
    T = rgb2gray(T);
    
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

    
    %% Find the best match [row column] using Normalized Cross Correlation (NCC)
    [NCCrow, NCCcol] = NCC(grayImage, T, NCC_THRESH);
    
    % If the best match exists
    % overlay the card name on the best match location on the NCC output image                      
    % Insert the card name on the output images (use small font size, e.g. 6)
    % set the overlay locations to the best match locations, plus-minus a random integer   

        
    
end

%% Display the output images 


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
Gray_row = size(grayImage,1);
Gray_col = size(grayImage,2);
SSD_matrix = zeros(Gray_row, Gray_col);
% assume that the template is within the image bounds (reasonable?)
for row = 1:(Gray_row)
    for col = 1:(Gray_col)
        row_offset = 0;
        col_offset = 0;
        squared_diff = (T - grayImage(row:row+T_row-1, col:col+T_col-1)).^2;
        ssd = sum(sum(squared_diff));
        SSD_matrix(row, col) = ssd;
    end
end
    disp(mean(mean(SSD_matrix)));
    disp(max(max(SSD_matrix)));
    disp(min(min(SSD_matrix)));
    imshow(SSD_matrix);

end

%% Implement the NCC-based template matching here
function [NCCrow, NCCcol] = NCC(grayImage, T, NCC_THRESH)
% inputs
%           grayImag        gray-scale image
%           T               gray-scale template
%           NCC_THRESH      threshold above which a match is accepted
% outputs
%           NCCrow          row of the best match (empty if unavailable)
%           NCCcol          column of the best match (empty if unavailable)


end
