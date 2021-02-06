function [ pxCost ] = pixelwiseAccuracy( prediction, groundtruth, sz)
%PIXELWISEACCURACY Summary of this function goes here
%   Detailed explanation goes here
labset1 = unique(prediction(:));
labset2 = unique(groundtruth(:));

[h, w] = size(groundtruth);
prediction = imresize(prediction, [h, w], 'nearest');

dismat = zeros(length(labset1), length(labset2));
for m = 1:length(labset1)
    for n = 1:length(labset2)
        consist = prediction==labset1(m) & groundtruth==labset2(n);
        dismat(m,n) = sum(consist(:));
    end
end

[Matching,Cost] = Hungarian(-dismat);
score = -Cost;

pxCost = score/prod(sz);
end

