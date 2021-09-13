% Multiple View Geometry 2021 semester B
% 
% Student: Itai Antebi, 204817498

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Solution for exercise #1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
clc;

%% question 2

% Load and plot the image
img2 = imread('compEx2.JPG ');
figure(); 
colormap gray;
imagesc(img2);
% Load points from file
load('compEx2.mat');
% Plot the points in the same figure as the image
hold on; 
plot(p1(1 ,:), p1(2 ,:), '.', 'MarkerSize', 20);
plot(p2(1 ,:), p2(2 ,:), '.', 'MarkerSize', 20);
plot(p3(1 ,:), p3(2 ,:), '.', 'MarkerSize', 20);
% Compute the lines going through the points
l1 = cross(p1(:,1), p1(:,2)); 
l2 = cross(p2(:,1), p2(:,2)); 
l3 = cross(p3(:,1), p3(:,2)); 
% Use the function rital to plot the lines in the same image
% 
% The lines indeed seem to be parallel in 3D
rital(l1);
rital(l2);
rital(l3);
% Compute the point of intersection between the second and third line and
% Plot this point in the same image
p23 = pflat(cross(l2, l3)); 
plot(p23(1) ,p23(2) , '.', 'MarkerSize', 20);
% Compute the distance between the first line and the the intersection point.
% 
% The distance in ~8 pixels. It is pretty small. 
% The three lines are parralel to each other in 3D so their projection onto
% the image plane should intersect at a single point, but noise and error
% causes this distance to be different than 0 (yet still small). 
distance_l1_p23 = abs(dot(l1, p23)) / sqrt(l1(1)^2 + l1(2)^2); 
display_distance_str = sprintf('The Distance Between l1 and the intersection of l2 and l3 is: %d pixels', distance_l1_p23);
disp(display_distance_str);
