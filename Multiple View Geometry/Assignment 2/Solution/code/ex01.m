% Multiple View Geometry 2021 semester B
% 
% Student: Itai Antebi, 204817498

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Solution for exercise #2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% question 1

% Load data
load('compEx1data.mat');

%% a.
% Plot the 3D points of the reconstruction & the cameras in the same figure.
figure(); 
plot3(X(1,:), X(2,:), X(3,:), '.', 'Markersize', 2);
hold on; 
plotcams(P); 
axis equal;
% This doesn't look like a reasonable reconstruction. For example, the
% walls do not meet at a normal 90-degrees angle. 

%% b. 
% Project the 3D points onto the first camera.
xproj = pflat(P{1} * X);
% Plot the image
img1 = imread(imfiles{1});
figure(); 
colormap gray;
imagesc(img1);
% Plot the projected points
visible = isfinite(x{1}(1 ,:));
hold on; 
plot(xproj(1,visible), xproj(2,visible), 'ro');
% Plot the image points
plot(x{1}(1,visible), x{1}(2,visible), 'b*');
% The projections DO appear to be close to the corresponding image points

%% c. 
% Modify all the 3D points and cameras using T1, T2
modified_X_by_T1 = pflat(T1 * X); 
modified_X_by_T2 = pflat(T2 * X); 
modified_P_by_T1 = cellfun(@(P) P / T1, P, 'UniformOutput', false);
modified_P_by_T2 = cellfun(@(P) P / T2, P, 'UniformOutput', false);
% Plot all the 3D points and cameras for each of the solutions
% solution 1:
figure(); 
plot3(modified_X_by_T1(1,:), modified_X_by_T1(2,:), modified_X_by_T1(3,:), '.', 'Markersize', 2);
hold on; 
plotcams(modified_P_by_T1); 
axis equal;
% solution 2:
figure(); 
plot3(modified_X_by_T2(1,:), modified_X_by_T2(2,:), modified_X_by_T2(3,:), '.', 'Markersize', 2);
hold on; 
plotcams(modified_P_by_T2); 
axis equal;
% The second solution looks very reasonable. For example, the walls meet at
% a normal 90-degrees angle. 

%% d. 
% Project the new 3D points obtained from using T1 into the first camera
projected_points_T1_1 = pflat(modified_P_by_T1{1} * modified_X_by_T1);
% Plot the image
figure(); 
colormap gray;
imagesc(img1);
% Plot the projected points
visible = isfinite(x{1}(1 ,:));
hold on; 
plot(projected_points_T1_1(1,visible), projected_points_T1_1(2,visible), 'ro');
% Plot the image points
plot(x{1}(1,visible), x{1}(2,visible ), 'b*');
