%% Compile the mex files
% Compile;

%% Set the image name
imgName = {'100_0109.png', 'b4nature_animals_land009.png', 'cheeky_penguin.png'};
gtName = '100_0109_groundtruth.png';

%% Set the output file name
outName = {'100_0109_rect.txt', 'b4nature_animals_land009_rect.txt', 'cheeky_penguin_rect.txt'};

for i=1:3
%% Load the input image
img = imread(imgName{i});

%% Load the ground truth image
% gtMask = imread(gtName);

%% Prompt the user to draw a rectangle over the object
imshow(img);
rect = getrect;

dlmwrite(outName{i}, rect, 'delimiter', '\t', 'precision', 3);
end
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

