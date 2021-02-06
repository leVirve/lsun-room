function [ result ] = algorithmFunc( img )
%ALGORITHMFUNC RETURN ROOM LAYOUT RESULT GIVEN IMAGE
%   Room layout should contain 4 field
%   type: room layout type
%   point: location of room corners
%   resolution: [w h] of the image
%   layout: the segmentation mask

[h,w,~] = size(img);
result.type = 0;
result.point = [[0.25; 0; 0.25; 0;0.75;1;0.75;1]*w [0.25; 0; 0.75; 1; 0.75;1;0.25;0]*h];
result.resolution = [h w];
result.layout  = getSegmentation( result );

end

