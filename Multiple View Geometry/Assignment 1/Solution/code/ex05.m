% Multiple View Geometry 2021 semester B
% 
% Student: Itai Antebi, 204817498

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Solution for exercise #1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
clc;

%% question 5

% Load the data and definitions from question
load('compEx5.mat');
M1 = eye(3); 
t1 = [0 0 0]'; 
P1 = [M1 t1];
% Plot the corner points and the image in the same 2D-figure.
% The origin is located at the top-left corner of the poster. 
img5 = imread('compEx5.JPG ');
figure(); 
colormap gray;
imagesc(img5);
axis equal; 
hold on; 
plot(corners(1 , [1:end 1]), corners(2 ,[1:end 1]), '*-');
% Plot normalized corners in a new 2D-figure
% The origin is located at the top-left corner of the poster, thanks to
% 'axis ij', but the normalized corners are locates around the origin in a
% very small scale
normalized_corners = K \ corners;
figure(); 
colormap gray;
imagesc(img5);
axis ij; 
axis equal; 
hold on; 
plot(normalized_corners(1 ,:), normalized_corners(2 ,:), '.', 'MarkerSize', 20);
% Compute the 3D points in the plane v that project onto the corner points
plane = pflat(v); 
s = - plane(1:end-1)' * normalized_corners; 
u_s = [normalized_corners ; s];
homogeneous_u_s = pflat(u_s); 
u_s_3D = homogeneous_u_s(1:3, :); 
% Compute the camera center and principal axis
center1 = - inv(M1) * t1;
direction1 = det(M1) * (M1(3,:))';
% Plot the 3D points together with the camera center and principal axis
% It looks reasonable
figure(); 
plot3(u_s_3D(1,:), u_s_3D(2,:), u_s_3D(3,:), '.', 'MarkerSize', 20);
axis ij; 
axis equal; 
hold on; 
plot3(center1(1), center1(2), center1(3), '.', 'MarkerSize', 20);
quiver3(center1(1), center1(2), center1(3), direction1(1), direction1(2), direction1(3), 1/norm(direction1));
% Compute the new camera
M2 = [sqrt(3)/2 0 0.5; 0 1 0; -0.5 0 sqrt(3)/2];
center2 = [2 0 0]'; 
direction2 = det(M2) * (M2(3,:))';
t2 = - M2 * center2; 
P2 = [M2 t2];
% Plot the new camera in the same figure
plot3(center2(1), center2(2), center2(3), '.', 'MarkerSize', 20);
quiver3(center2(1), center2(2), center2(3), direction2(1), direction2(2), direction2(3), 1/norm(direction2));
% Compute the homography and transform the normalized corner points to the
% new (virtual)image.
H = M2 - t2 * plane(1:end-1)';
homogeneous_u_s_3D_camera_2 = pflat(H * u_s_3D); 
% Plot the transformed points in a new 2D-figure.
% The result does look like I would expect it when moving the camera like
% this. 
figure(); 
plot(homogeneous_u_s_3D_camera_2(1,:), homogeneous_u_s_3D_camera_2(2,:), '.', 'MarkerSize', 20);
axis ij; 
axis equal; 
hold on; 
% Project the 3D points into the same image using the camera matrix
% They are projected to the exact same location - same result
u_s_projected_on_camera_2 = pflat(P2 * u_s); 
plot(u_s_projected_on_camera_2(1,:), u_s_projected_on_camera_2(2,:), '.', 'MarkerSize', 20);
% Transform the original image and the corner points using the homography Htot
H_tot = K * H / K; 
tform = maketform('projective', H_tot');
[new_img5, xdata, ydata] = imtransform(img5, tform, 'size', size(img5));
transformed_corners = pflat(H_tot * corners); 
% Plot both in a new 2D-figure
figure(); 
colormap gray; 
imagesc(xdata, ydata, new_img5);
axis ij; 
axis equal; 
hold on; 
plot(transformed_corners(1,:), transformed_corners(2,:), '.', 'MarkerSize', 20);
