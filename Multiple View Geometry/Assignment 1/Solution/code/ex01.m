% Multiple View Geometry 2021 semester B
% 
% Student: Itai Antebi, 204817498

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Solution for exercise #1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% question 1

% Load points from file
load('compEx1.mat');
% Divide the homogeneous coordinates with their last entry
homogeneous_x2D = pflat(x2D); 
homogeneous_x3D = pflat(x3D); 
% Plot the result
figure(); 
plot(homogeneous_x2D(1,:), homogeneous_x2D(2,:), '.');
figure(); 
plot3(homogeneous_x3D(1,:), homogeneous_x3D(2,:), homogeneous_x3D(3,:), '.');
axis equal;
