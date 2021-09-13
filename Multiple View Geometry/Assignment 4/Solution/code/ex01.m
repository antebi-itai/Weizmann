% Multiple View Geometry 2021 semester B
% 
% Student: Itai Antebi, 204817498

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Solution for exercise #4
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% question 1

% load the data
load('compEx1data.mat');

%% a
% Solve the total least squares problem with all the points and find the plane which contains the wall
homogeneous_X = pflat(X);
plane = closest_plane(homogeneous_X);
% Compute the RMS distance between the 3D-points and the plane
RMS_of_plane = sqrt(mean((plane' * homogeneous_X) .^ 2));
fprintf('RMS distance between all 3D points and least squares plane = %f\n\n', RMS_of_plane);

%% b
% Use RANSAC to robustly fit a plane to the 3D points X
num_of_points = size(X, 2);
N = 6;
thresh = 0.1;

max_inliers = 0;
best_ransac_inliers = [];
best_ransac_plane = [];
for i = 1:N
    % choose 3 random points
    randind = randi([1, num_of_points],1,3);
    % compute the plane according to these points
    plane = null(homogeneous_X(:, randind)');
    plane = plane ./ norm(plane(1:3));
    % count the number of points that agree with this plane
    inliers = abs(plane' * homogeneous_X) <= thresh;
    num_inliers = sum(inliers);
    if (num_inliers > max_inliers)
        max_inliers = num_inliers;
        best_ransac_inliers = inliers;
        best_ransac_plane = plane;
    end
end

% How many inliers do you get?
fprintf('Number of inliers at RANSAC model = %d\n', max_inliers);
% Compute the RMS distance between the plane obtained with RANSAC to the 3D points
RMS_of_RANSAC_plane = sqrt(mean((best_ransac_plane' * homogeneous_X(:, best_ransac_inliers)) .^ 2));
fprintf('RMS distance between inlier points and RANSAC plane = %f\n\n', RMS_of_RANSAC_plane);
% Plot the absolute distances between the plane and the points in a histogram with 100 bins.
ransac_dists = abs(best_ransac_plane' * X);
figure;
hist(ransac_dists ,100);

%% c
% Solve the total least squares problem with only the inliers.
inliers_plane  = closest_plane(homogeneous_X(:, best_ransac_inliers));
% Compute the RMS distance between the 3D-points and the plane
RMS_of_inliers_plane = sqrt(mean((inliers_plane' * homogeneous_X(:, best_ransac_inliers)) .^ 2));
fprintf('RMS distance between inlier points and least squares inliers plane = %f\n\n', RMS_of_inliers_plane);
% Plot the absolute distances between the plane and the points in a histogram with 100 bins.
inliers_dists = abs(inliers_plane' * X);
figure;
hist(inliers_dists ,100);

% Plot the projection of the inliers into the images. Where are they located?
homogeneous_inlier = homogeneous_X(:, best_ransac_inliers);
x1_inlier = pflat(P{1} * homogeneous_inlier);
x2_inlier = pflat(P{2} * homogeneous_inlier);
im1 = imread("./house1.jpg");
im2 = imread("./house2.jpg");

figure;
hold on;
imagesc(im1);
plot(x1_inlier(1 , :) ,x1_inlier(2, :) , 'r.', "MarkerSize", 20);
axis equal;
axis ij;

figure;
hold on;
imagesc(im2);
plot(x2_inlier(1 , :) ,x2_inlier(2, :) , 'r.', "MarkerSize", 20);
axis equal;
axis ij;

%% d
% compute a homography from camera 1 to camera 2
P2_normalized = inv(K) * P{2};
R = P2_normalized(:, 1:3); 
t = P2_normalized(:, 4); 
pi = pflat(inliers_plane); 
H = R - t * pi(1:3)'; 

% Plot the points x in image 1
figure;
hold on;
imagesc(im1);
plot(x(1 , :) ,x(2, :) , 'r.', "MarkerSize", 20);
axis equal;
axis ij;
% Transform the points using the homography
x_normalized = pflat(inv(K) * x);
x_homographied_normalized = pflat(H * x_normalized);
x_homographied = K * x_homographied_normalized;
% Plot the points x in image 2
figure;
colormap gray;
hold on;
imagesc(im2);
plot(x_homographied(1 , :) , x_homographied(2, :) , 'r.', "MarkerSize", 20);
axis equal;
axis ij;