function [ meanPtError, allPtError, meanPxError, allPxError ] = evaluationFunc( result, data )
%EVALUATIONFUNC EVALUATE PIXELWISE ACCURACY AND CORNERWISE ACCURACY
%   result: prediciton
%   data: ground truth
GlobalParameters;
allPtError = zeros(length(result),1);
allPxError = zeros(length(result),1);

for i = 1:length(result)
    if isfield(result(i), 'point');
        allPtError(i) = cornerError(result{i}.point, data(i).point, data(i).resolution);
    else
        allPtError(i) = nan;
    end
    layout_path = sprintf(LAYOUT_PATTERN,data(i).image);
    if exist(layout_path, 'file')
        load(layout_path);
        layout = imread(sprintf('../data/layout_seg_images/%s.png',data(i).image));
        allPxError(i) = 1 - pixelwiseAccuracy(result{i}.layout, layout, data(i).resolution);
    else
        allPxError(i) = nan;
    end
end
meanPtError = mean(allPtError);
meanPxError = mean(allPxError);

end

