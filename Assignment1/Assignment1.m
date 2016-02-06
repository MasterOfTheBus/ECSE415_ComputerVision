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
SSD_THRESH = 50000;
NCC_THRESH = 50000;

%% Initialize two output images to the RGB input image
output_img1 = image;
output_img2 = image;

%% For each template, do the following
for i=1:numTemplates
    %% Load the RGB template image, into variable T
    filepath = sprintf('Template images/%s', templateFileNames(i).name);
    T = imread(filepath);
    
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
half_Tr = idivide(T_row, int32(2));
half_Tc = idivide(T_col, int32(2));
Gray_row = size(grayImage,1);
Gray_col = size(grayImage,2);
SSD_matrix = zeros(Gray_row-T_row, Gray_col-T_col);
min_ssd = realmax;
min_row = 0;
min_col = 0;
% assume that the template is within the image bounds (reasonable?)
% for row = 1:(Gray_row)
%     for col = 1:(Gray_col)
%         % make sure that image size is not exceeded
%         % adjust template size for boundaries
%         TR = T_row;
%         TC = T_col;
%         row_offset = row + T_row-1;
%         col_offset = col + T_col-1;
%         if (row + T_row-1) > Gray_row
%             row_offset = Gray_row;
%             TR = T_row - (row+T_row-1-Gray_row);
%         end
%         if (col + T_col-1) > Gray_col
%             col_offset = Gray_col;
%             TC = T_col - (col+T_col-1-Gray_col);
%         end
%         
%         squared_diff = (T(1:TR, 1:TC) - grayImage(row:row_offset, col:col_offset)).^2;
%         ssd = sum(sum(squared_diff));
%         SSD_matrix(row, col) = ssd;
%     end
% end
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
        SSD_matrix(row-half_Tr+1, col-half_Tc+1) = ssd;
    end
end

    disp(mean(mean(SSD_matrix)));
    disp(max(max(SSD_matrix)));
    k = min(min(SSD_matrix));
    index = find(SSD_matrix==k);
    if (k < SSD_THRESH)
       disp(k);
       disp(index);
    end
    SSDrow = -1;
    SSDcol = -1;
    if (min_ssd < SSD_THRESH)
       disp(min_ssd);
       disp(min_row);
       disp(min_col);
       
       SSDrow = min_row;
       SSDcol = min_col;
    end
%     imshow(SSD_matrix);

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
