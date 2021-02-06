function [ result ] = predictFunc( data )
%ALGORITHM PERFORM PREDICTION FOR A BATCH OF DATA
%   data can be testing or validation

GlobalParameters;
result = cell(length(data),1);

PREDCTION_OUTPUT_PATTERN = '../output/super_eval/fcn32s_adam_404(+L1)-weights.08.hdf5/%s.png';

for i = 1:length(data)
    result{i}.layout = imread(sprintf(PREDCTION_OUTPUT_PATTERN, data(i).image));

%     img = imread(sprintf(IMAGE_PATTERN, data(i).image));
    % a fake method
%     result{i} = algorithmFunc(img);
    % a cheating method
%     tmp = data(i);
%     tmp.layout = getSegmentation(data(i));
%     result{i} = tmp;

end

