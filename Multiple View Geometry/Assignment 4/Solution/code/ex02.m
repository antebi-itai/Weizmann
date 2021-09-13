% Multiple View Geometry 2021 semester B
% 
% Student: Itai Antebi, 204817498

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Solution for exercise #4
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% question 2

% load the data
load('compEx2data.mat');

% RANSAC
num_of_points = size(xA, 2); 
homogeneous_xA = [xA ; ones(1,num_of_points)];
homogeneous_xB = [xB ; ones(1,num_of_points)];
N = 100;
thresh = 5;

max_inliers = 0;
best_ransac_inliers = [];
best_ransac_homography = [];
for i = 1:N
    % choose 4 random correspondences
    randind = randi([1, num_of_points],1,4);
    x1 = homogeneous_xA(:, randind);
    x2 = homogeneous_xB(:, randind);
    % compute the homography according to these points, using DLT
    M = zeros(3 * 4, 9 + 4);
    for j=1:4
        M((j-1)*3+1, 1:3) = x1(:, j)';
        M((j-1)*3+2, 4:6) = x1(:, j)';
        M((j-1)*3+3, 7:9) = x1(:, j)';
        M((j-1)*3+1, 9+j) = -x2(1,j);
        M((j-1)*3+2, 9+j) = -x2(2,j);
        M((j-1)*3+3, 9+j) = -x2(3,j);
    end
    [U, S, V] = svd(M);
    v = V(:,end);
    H = reshape(v(1:9), [3, 3])';
    % count the number of points that agree with this plane
    diff = pflat(H * homogeneous_xA) - homogeneous_xB; 
    inliers = sqrt(sum(diff.^2)) < thresh; 
    num_inliers = sum(inliers);
    if (num_inliers > max_inliers)
        max_inliers = num_inliers;
        best_ransac_inliers = inliers;
        best_ransac_homography = H;
    end
end

% How many inliers do you get?
fprintf('Number of inliers at RANSAC model = %d\n', max_inliers);

% compute the homography according to inlier points, using DLT
x1 = homogeneous_xA(:, best_ransac_inliers);
x2 = homogeneous_xB(:, best_ransac_inliers);
M = zeros(3 * max_inliers, 9 + max_inliers);
for j=1:max_inliers
    M((j-1)*3+1, 1:3) = x1(:, j)';
    M((j-1)*3+2, 4:6) = x1(:, j)';
    M((j-1)*3+3, 7:9) = x1(:, j)';
    M((j-1)*3+1, 9+j) = -x2(1,j);
    M((j-1)*3+2, 9+j) = -x2(2,j);
    M((j-1)*3+3, 9+j) = -x2(3,j);
end
[U, S, V] = svd(M);
v = V(:,end);
bestH = reshape(v(1:9), [3, 3])';

% Plot
A = imread('a.JPG');
B = imread('b.JPG');
tform = maketform('projective', bestH');
transfbounds = findbounds(tform ,[1, 1; size(A,2), size(A,1)]);
xdata = [min([transfbounds(: ,1); 1]) max([transfbounds(: ,1); size(B ,2)])];
ydata = [min([transfbounds(: ,2); 1]) max([transfbounds(: ,2); size(B ,1)])];
[ newA ] = imtransform(A, tform, 'xdata', xdata, 'ydata',ydata);
tform2 = maketform('projective', eye(3));
[ newB ] = imtransform(B, tform2, 'xdata', xdata, 'ydata', ydata, 'size', size(newA));
newAB = newB;
newAB( newB < newA ) = newA( newB < newA );
figure;
imshow(newAB);