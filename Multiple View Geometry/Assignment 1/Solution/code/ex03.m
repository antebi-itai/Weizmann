% Multiple View Geometry 2021 semester B
% 
% Student: Itai Antebi, 204817498

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Solution for exercise #1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
clc;

%% question 3

% Load the startpoints and endpoints
load('compEx3.mat');
% Define H1 H2 H3 H4
H1 = [sqrt(3) -1 1; 1 sqrt(3) 1; 0 0 2]; 
H2 = [1 -1 1; 1 1 0; 0 0 1];
H3 = [1 1 0; 0 2 0; 0 0 1]; 
H4 = [sqrt(3) -1 1; 1 sqrt(3) 1; 0.25 0.5 2];
% Move points to their homogenous representation
homogeneous_startpoints = [startpoints; ones(1, size(startpoints, 2))];
homogeneous_endpoints = [endpoints; ones(1, size(endpoints, 2))];
% Compute the transformations of the given start and endpoints
H1_startpoints = pflat(H1 * homogeneous_startpoints); 
H1_endpoints = pflat(H1 * homogeneous_endpoints); 
H2_startpoints = pflat(H2 * homogeneous_startpoints); 
H2_endpoints = pflat(H2 * homogeneous_endpoints); 
H3_startpoints = pflat(H3 * homogeneous_startpoints); 
H3_endpoints = pflat(H3 * homogeneous_endpoints); 
H4_startpoints = pflat(H4 * homogeneous_startpoints); 
H4_endpoints = pflat(H4 * homogeneous_endpoints); 
% Plot the transformed lines
figure(); 
plot([H1_startpoints(1,:); H1_endpoints(1,:)], [H1_startpoints(2,:); H1_endpoints(2,:)], 'r-');
hold on;
plot([H2_startpoints(1,:); H2_endpoints(1,:)], [H2_startpoints(2,:); H2_endpoints(2,:)], 'g-');
plot([H3_startpoints(1,:); H3_endpoints(1,:)], [H3_startpoints(2,:); H3_endpoints(2,:)], 'b-');
plot([H4_startpoints(1,:); H4_endpoints(1,:)], [H4_startpoints(2,:); H4_endpoints(2,:)], 'k-');
axis equal;
% Properties:
% H1 - preserves lengths between points
% H1, H2 - preserves angles between lines
% H1, H2, H3 - map parallel lines to paralel lines
% Classification:
% H1 - Euclidian
% H2 - Similarity
% H3 - Affine
% H4 - Projective Transformation
