function [ ptCost ] = cornerError( prediction, groundtruth, sz )
%CORNERACCURACY Summary of this function goes here
%   Detailed explanation goes here
distmat = pdist2( prediction, groundtruth);
[Matching,Cost] = Hungarian(distmat);

ptCost = sum(Cost./norm(sz)) + abs(size(prediction,1)-size(groundtruth,1))*1/3;
ptCost = ptCost./max( size(prediction,1), size(groundtruth,1));
end

