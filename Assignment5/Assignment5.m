%% Load the input video
vidReader = VideoReader('flow.avi');

%% Create optical flow objects
lkFlow = opticalFlowLK();
hsFlow = opticalFlowHS();

figure('Position', [10 10 1000 500]);

%% do for each video frame
while hasFrame(vidReader)
    % read a video frame
    frame = readFrame(vidReader);
%     frame = imresize(frame, 3);
    
    % estimate the LK-based motion field
    flow1 = estimateFlow(lkFlow, frame);
    
    % estimate the HS-based motion field 
    flow2 = estimateFlow(hsFlow, frame);
    
    % display the LK optical flow 
    subplot(1,2,1);
    imshow(frame);
    hold on
    plot(flow1, 'ScaleFactor', 2); % scaling makes it too large
    
    % display the HS optical flow
    subplot(1,2,2);
    imshow(frame);
    hold on
    plot(flow2, 'ScaleFactor', 2);
    
    % pause execution (helps in updating the subplots)
    pause(0)
    
    hold off
end