% Multiple View Geometry 2021 semester B
% 
% Student: Itai Antebi, 204817498

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Solution for exercise #1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
clc;

%% question 4

% Load the data and plot the images
load('compEx4.mat');
homogeneous_U = pflat(U); 
img4_1 = imread('compEx4im1.JPG ');
img4_2 = imread('compEx4im2.JPG ');
% Compute the camera centers and principal axes of the cameras
M1 = P1(1:3, 1:3);
M2 = P2(1:3, 1:3);
p41 = P1(:, 4);
p42 = P2(:, 4);
center1 = - inv(M1) * p41;
center2 = - inv(M2) * p42;
direction1 = det(M1) * (M1(3,:))';
direction2 = det(M2) * (M2(3,:))';
% Plot the 3D-points in U and the camera centers and a vector in the
% direction of the principal axes
figure(); 
plot3(homogeneous_U(1,:), homogeneous_U(2,:), homogeneous_U(3,:), '.');
hold on; 
plot3(center1(1), center1(2), center1(3), '.', 'MarkerSize', 20);
plot3(center2(1) ,center2(2) , center2(3), '.', 'MarkerSize', 20);
quiver3(center1(1), center1(2), center1(3), direction1(1), direction1(2), direction1(3), 3/norm(direction1));
quiver3(center2(1), center2(2), center2(3), direction2(1), direction2(2), direction2(3), 3/norm(direction2));
axis equal; 
% Project the points in U into the cameras P1 and P2
% The results does look reasonable
homogeneous_U_P1 = pflat(P1 * U); 
homogeneous_U_P2 = pflat(P2 * U); 
% cameras P1
figure(); 
colormap gray;
imagesc(img4_1);
hold on; 
plot(homogeneous_U_P1(1,:), homogeneous_U_P1(2,:), '.', 'MarkerSize', 20);
% cameras P2
figure(); 
colormap gray;
imagesc(img4_2);
hold on; 
plot(homogeneous_U_P2(1,:), homogeneous_U_P2(2,:), '.', 'MarkerSize', 20);
