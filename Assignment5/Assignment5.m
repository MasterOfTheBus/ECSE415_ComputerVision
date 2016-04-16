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
    
    % estimate the LK-based motion field
    flow1 = estimateFlow(lkFlow, frame);
    
    % estimate the HS-based motion field 
    flow2 = estimateFlow(hsFlow, frame);
    
    % display the LK optical flow 
    subplot(1,2,1);
    imshow(frame);
    hold on
    plot(flow1, 'DecimationFactor', [10 10], 'ScaleFactor', 10);
    
    % display the HS optical flow
    subplot(1,2,2);
    imshow(frame);
    hold on
    plot(flow2, 'DecimationFactor', [10 10], 'ScaleFactor', 35);
    
    % pause execution (helps in updating the subplots)
    pause(0)
    
    hold off
end