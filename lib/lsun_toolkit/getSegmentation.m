function [ segment ] = getSegmentation( data )
%GETSEGMENTATION CONVERT ROOM LAYOUT TO SEGMENTATION MASK
%   data.type: room layout type
%   data.point: room corners position
%   data.resolution: [w h] of the image
if data.type==11
    segment = [];
    return;
end

type = [];
RoomLayoutTypes;

point = data.point;
RecordId = find([type.typeid] == data.type);

lines = type(RecordId).lines;

pt1s = round([point(lines(:,1),2) point(lines(:,1),1)]); 
pt1s(pt1s(:)<=0) = 1;
pt1s(pt1s(:,1)>data.resolution(1), 1)  = data.resolution(1);
pt1s(pt1s(:,2)>data.resolution(2), 2)  = data.resolution(2);
pt2s = round([point(lines(:,2),2) point(lines(:,2),1)]); 
pt2s(pt2s(:)<=0) = 1;
pt2s(pt2s(:,1)>data.resolution(1), 1)  = data.resolution(1);
pt2s(pt2s(:,2)>data.resolution(2), 2)  = data.resolution(2);

lineplot = drawLineOnMatrix(data.resolution, pt1s, pt2s);
CC = bwconncomp(1-lineplot, 4);

% I = cellfun('isempty',type(RecordId).region);
% numRegion = sum(~I);
% assert(CC.NumObjects == numRegion);
segment = zeros(data.resolution);
for i = 1:length(CC.PixelIdxList)
    segment(CC.PixelIdxList{i}) = i;
end

se = strel('disk',2);
segment = imclose(segment, se);
assert(sum(segment(:)==0)==0);
end

