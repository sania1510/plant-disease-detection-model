zzzzzzz% Load the trained model
load('plant_disease_net.mat', 'trainedNet');

% Specify the path to the test image
newImagePath = 'E:\MATLAB\test_image_3.jpg'; 
img = imread(newImagePath);

% Check if the image is grayscale and convert to RGB if needed
if size(img, 3) == 1
    img = cat(3, img, img, img); % Convert grayscale to RGB
end

% Resize the image to the input size of the network
inputSize = [227, 227, 3];
preprocessedImg = imresize(img, inputSize(1:2));

% Segment the disease region using K-means clustering
segmentedImg = segmentDisease(img);

% Classify the preprocessed image
[predictedLabel, scores] = classify(trainedNet, preprocessedImg);

% Display the results
disp(['Predicted Label: ', char(predictedLabel)]);
disp('Scores for each class:');
disp(scores);

% Format the label to avoid issues with underscores
formattedLabel = strrep(char(predictedLabel), '_', ' ');

% Display the original image with the predicted label
figure;
subplot(1, 2, 1);
imshow(img);
title(['Predicted: ', formattedLabel]);

% Display the segmented image
subplot(1, 2, 2);
imshow(segmentedImg);
title('Segmented Image');

% Function to segment disease region
function segmentedImg = segmentDisease(img)
    % Convert RGB to LAB color space
    labImg = rgb2lab(img);
    
    % Extract the a* and b* channels
    ab = labImg(:, :, 2:3);
    ab = im2single(ab);
    
    % Perform K-means clustering
    nColors = 3; % Number of clusters
    pixelLabels = imsegkmeans(ab, nColors, 'NumAttempts', 3);
    
    % Convert the cluster labels to an RGB image
    segmentedImg = label2rgb(pixelLabels);
end
