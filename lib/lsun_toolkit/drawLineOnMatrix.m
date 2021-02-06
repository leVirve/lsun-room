function [ matrix ] = drawLineOnMatrix( sz, pt1s, pt2s )
%DRAWLINEONMATRIX DRAW LINES DEFINED BY TWO POINTS ON A MATRIX
%   sz: size of matrix
%   pt1s, pt2s: [x,y] of two points

matrix = zeros(sz(1), sz(2));
for i = 1:size(pt1s,1)
    pt1 = pt1s(i,:);
    pt2 = pt2s(i,:);
    
    if pt1(1)>pt2(1)
        tmp = pt1;
        pt1 = pt2;
        pt2 = tmp;
    end
    for j = pt1(1):pt2(1)
        y = j;
        x = pt1(2) + (pt2(2)-pt1(2))*(y-pt1(1))/(pt2(1)-pt1(1)+0.00001);
        matrix(y,round(x)) = 1;
    end
    if pt1(2)>pt2(2)
        tmp = pt1;
        pt1 = pt2;
        pt2 = tmp;
    end
    for j = pt1(2):pt2(2)
        x = j;
        y = pt1(1) + (pt2(1)-pt1(1))*(x-pt1(2))/(pt2(2)-pt1(2)+0.00001);
        matrix(round(y),x) = 1;
    end
end



end

