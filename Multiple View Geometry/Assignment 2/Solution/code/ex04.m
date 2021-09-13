% Multiple View Geometry 2021 semester B
% 
% Student: Itai Antebi, 204817498

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Solution for exercise #2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% question 4

% Load data
load('cube_matches.mat'); 
img1 = imread('cube1.JPG');
img2 = imread('cube2.JPG');

%% a. 
% Set up the DLT equations for triangulation, and solve the homogeneous least squares system.
n = size(x1, 2);
X_3D = zeros(4, n);
lambdas = zeros(2, n);

for i = 1:n
    x1_i = [x1(:, i); 1]; 
    x2_i = [x2(:, i); 1]; 
    
    M_i = zeros(6, 6);
    M_i(1:3, 1:4) = P1;
    M_i(4:6, 1:4) = P2;
    M_i(1:3,5) = -x1_i;
    M_i(4:6,6) = -x2_i;
    
    [Ui, Si, Vi] = svd(M_i);
    Xi = pflat(Vi(1:4, 6));
    lambda_i = Vi(5:6, 6);
    X_3D(:, i) = Xi;
    lambdas(:, i) = lambda_i; 
end

% Project the computed points back into the two images
x_2D_P1 = pflat(P1 * X_3D); 
x_2D_P2 = pflat(P2 * X_3D); 

% compare with the corresponding SIFT-points x1 and x2
% Visualize
figure;
%imshow(img1); 
hold on;
plot(x1(1 ,:), x1(2 ,:), 'bo');
plot(x_2D_P1(1 ,:), x_2D_P1(2 ,:), 'r*');
%axis ij; 
axis equal;
figure;
%imshow(img2); 
hold on;
plot(x2(1 ,:), x2(2 ,:), 'bo');
plot(x_2D_P2(1 ,:), x_2D_P2(2 ,:), 'r*');
%axis ij; 
axis equal;

% Compare with the results you get when you normalize with inner parameters of the cameras
N_x1 = pflat(K1 \ [x1; ones(1, n)]);
N_x1 = N_x1(1:2, :); 
N_x2 = pflat(K2 \ [x2; ones(1, n)]);
N_x2 = N_x2(1:2, :); 
N_P1 = K1 \ P1;
N_P2 = K2 \ P2;
X_3D = zeros(4, n);
lambdas = zeros(2, n);
for i = 1:n
    x1_i = [N_x1(:, i); 1]; 
    x2_i = [N_x2(:, i); 1]; 
    M_i = zeros(6, 6);
    M_i(1:3, 1:4) = N_P1;
    M_i(4:6, 1:4) = N_P2;
    M_i(1:3,5) = -x1_i;
    M_i(4:6,6) = -x2_i;
    [Ui, Si, Vi] = svd(M_i);
    Xi = pflat(Vi(1:4, 6));
    lambda_i = Vi(5:6, 6);
    X_3D(:, i) = Xi;
    lambdas(:, i) = lambda_i; 
end
% Project the computed points back into the two images
x_2D_R1 = pflat(N_P1 * X_3D); 
x_2D_R2 = pflat(N_P2 * X_3D); 
% compare with the corresponding SIFT-points x1 and x2
% Visualize
figure;
hold on;
plot(N_x1(1 ,:), N_x1(2 ,:), 'bo');
plot(x_2D_R1(1 ,:), x_2D_R1(2 ,:), 'r*');
axis equal;
figure;
hold on;
plot(N_x2(1 ,:), N_x2(2 ,:), 'bo');
plot(x_2D_R2(1 ,:), x_2D_R2(2 ,:), 'r*');
axis equal;
% There is a small improvement when using the normalized points and
% matrices. For example, it is noticeable that there are less "fleeing"
% points. 

%% b. 

% Remove the points for which the error in at least one of the images is larger than 3 pixels
goodpoints = (sqrt(sum((x1 - x_2D_P1(1:2 ,:)).^2)) < 3) & (sqrt(sum((x2 - x_2D_P2(1:2 ,:)).^2)) < 3); 
good_3D = X_3D(:, goodpoints); 

% Plot the remaining 3D points, the cameras and the cube model in the same 3D plot.
cameras = {P1, P2};
figure;
hold on;
plot3(good_3D(1, :), good_3D(2, :), good_3D(3, :), 'ko');
plotcams(cameras);
plot3(hom_X(:, 1), hom_X(:, 2), hom_X(:, 3), 'bo');
axis equal;
view(-20,5)
