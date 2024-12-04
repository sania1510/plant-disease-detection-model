%% Step 1: Load and Preprocess the Dataset
clc; clear; close all;

% Specify the path to the dataset folder
datasetPath = 'E:\MATLAB\Plant Village Dataset\archive';

% Load the dataset using imageDatastore
imds = imageDatastore(datasetPath, ...
                      'IncludeSubfolders', true, ...
                      'LabelSource', 'foldernames');

% Display total number of images and their labels
disp(['Total Images: ', num2str(numel(imds.Files))]);
disp('Categories:');
disp(unique(imds.Labels));

% Split the dataset into training (80%) and testing (20%) sets
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.8, 'randomized');

% Check if imdsTrain is empty
if isempty(imdsTrain)
    error('Training dataset is empty. Check your dataset path and structure.');
end

% Check sample images
figure;
perm = randperm(numel(imdsTrain.Files), 9);
for i = 1:9
    subplot(3, 3, i);
    img = readimage(imdsTrain, perm(i));
    imshow(img);
    title(imdsTrain.Labels(perm(i)));
end

%% Step 2: K-means Clustering for Image Segmentation (Optional)
function segmentedImg = segmentDisease(img)
    labImg = rgb2lab(img);
    ab = labImg(:, :, 2:3);
    ab = im2single(ab);
    nColors = 3; % Set the number of clusters
    pixelLabels = imsegkmeans(ab, nColors, 'NumAttempts', 3);
    segmentedImg = label2rgb(pixelLabels);
end

% Test K-means Clustering on a sample image
sampleImg = readimage(imdsTrain, 1);
segmentedImg = segmentDisease(sampleImg);
figure; imshow(segmentedImg); title('Segmented Image');

%% Step 3: Build and Train a Deep Learning Model
% Load Pre-trained CNN (Transfer Learning using AlexNet)
try
    net = alexnet; 
catch
    error('AlexNet is not available.');
end

% Display network structure
disp('Loaded AlexNet Model:');
analyzeNetwork(net);

layers = net.Layers;

% Modify the last layers for your dataset's number of classes
numClasses = numel(categories(imdsTrain.Labels));
disp(['Number of Classes: ', num2str(numClasses)]);

layers(end-2) = fullyConnectedLayer(numClasses, 'Name', 'new_fc');
layers(end) = classificationLayer('Name', 'new_output');

% Verify modified layers
disp('Modified Layers:');
disp(layers(end-2:end));

% Resize images to 227x227 (required by AlexNet) using augmentedImageDatastore
inputSize = [227, 227, 3];
augImdsTrain = augmentedImageDatastore(inputSize, imdsTrain);
augImdsTest = augmentedImageDatastore(inputSize, imdsTest);

% Set training options
options = trainingOptions('sgdm', ...
                          'InitialLearnRate', 0.001, ...
                          'MaxEpochs', 10, ...
                          'MiniBatchSize', 32, ...
                          'ValidationData', augImdsTest, ...
                          'ValidationFrequency', 10, ...
                          'Verbose', true, ...
                          'Plots', 'training-progress');

% Train the network
disp('Training the Network...');
trainedNet = trainNetwork(augImdsTrain, layers, options);

% Save the trained model
save('plant_disease_net.mat', 'trainedNet');
disp('Training Complete. Model saved as plant_disease_net.mat');

%% Step 4: Evaluate the Model
% Load the trained model (if needed)
load('plant_disease_net.mat', 'trainedNet');

% Test the model on the test dataset
[predLabels, scores] = classify(trainedNet, augImdsTest);
accuracy = mean(predLabels == imdsTest.Labels);
disp(['Model Accuracy: ', num2str(accuracy * 100), '%']);

%%

% Plot the confusion matrix
figure;
plotconfusion(imdsTest.Labels, predLabels);

% Save the confusion matrix plot as an image (PNG format)
saveas(gcf, 'confusion_matrix.png');

% Alternatively, save with higher resolution using exportgraphics
exportgraphics(gcf, 'confusion_matrix_highres.png', 'Resolution', 300);

disp('Confusion matrix saved as confusion_matrix.png');

%%

[confMat, order] = confusionmat(imdsTest.Labels, predLabels);
precision = diag(confMat) ./ sum(confMat, 2);
recall = diag(confMat) ./ sum(confMat, 1)';
F1_score = 2 * (precision .* recall) ./ (precision + recall);
disp(table(order, precision, recall, F1_score));


