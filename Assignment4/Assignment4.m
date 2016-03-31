%% Compile the mex files
% Compile;

%% Set the image name
imgName = '100_0109.png';
gtName = '100_0109_groundtruth.png';

%% Set the output file name
outName = 'rect.txt';

%% Load the input image
img = imread(imgName);

%% Load the ground truth image
gtMask = imread(gtName);

%% Prompt the user to draw a rectangle over the object
imshow(img);
rect = getrect;

dlmwrite(outName, rect, 'delimiter', '\t', 'precision', 3);

% %% Clustering using K-means
% kmeansMask = kmeansMex(imgName,rect);
% 
% %% Clustering using GMM
% gmmMask = GmmMex(imgName,rect);
% 
% %% Clustering using GraphCut
% graphcutMask  = GraphCutMex(imgName,rect);
% 
% %% Display the segmentation results

