%% Load the input video
vidReader = VideoReader('flow.avi');

%% Create optical flow objects

%% do for each video frame
while hasFrame(vidReader)
    % read a video frame
    
    % estimate the LK-based motion field
    
    % estimate the HS-based motion field  

    % display the LK optical flow 
    subplot(1,2,1);
    
    % display the HS optical flow
    subplot(1,2,2);
    
    % pause execution (helps in updating the subplots)
    pause(0)
end