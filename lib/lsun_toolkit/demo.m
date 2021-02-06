% Demo: LSUN Room layout estimation

addpath(genpath('.'));
GlobalParameters;

%% Test your algorithm on validation set with ground truth
% On the evaluation server, we will call similar function like
% evaluationFunc and compare submissions with mean corner error and mean
% pixelwise error.

load(VALIDATION_DATA_PATH);
results = predictFunc(validation);
[ meanPtError, allPtError, meanPxError, allPxError ] = evaluationFunc(results, validation);
fprintf('Mean Corner Error: %3.2f\n', meanPtError);
fprintf('Mean Pixel Error: %3.2f\n', meanPxError);


%% What to submit?
% Run your algorithm, and submit your result as in "results". It contains:
%   type: room layout type
%   point: room corner [x y]
%   layout: room layout segmentation mask
%   if the last one is not included, we will automaticall fill in by
%   calling function "getSegmentation".

% load(TEST_DATA_PATH);
% results = predictFunc(testing(1:10));
% [ meanPtError, allPtError, meanPxError, allPxError ] = evaluationFunc(results, testing(1:10));
% fprintf('Mean Corner Error: %3.2f\n', meanPtError);
% fprintf('Mean Pixel Error: %3.2f\n', meanPxError);
