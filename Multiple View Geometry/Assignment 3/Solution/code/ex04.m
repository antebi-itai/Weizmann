% Multiple View Geometry 2021 semester B
% 
% Student: Itai Antebi, 204817498

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Solution for exercise #3
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% question 4

%% Using triangulation, find P2 and X s.t. most points are in front of both cameras
% Compute four camera solutions in equation 11
[U, S, V] = svd(E); 
u3 = U(:, 3); 
W = [0 -1 0; 1 0 0; 0 0 1]; 
P1 = [eye(3), zeros(3,1)]; 
P2_1 = [U*W*V' u3]; 
P2_2 = [U*W*V' -u3]; 
P2_3 = [U*W'*V' u3]; 
P2_4 = [U*W'*V' -u3]; 
% Normalize the points
x1_k_normalized = inv(K) * x{1}; 
x2_k_normalized = inv(K) * x{2}; 
mean_1 = [mean(x1_k_normalized(1:2 ,:) ,2); 0];
mean_2 = [mean(x2_k_normalized(1:2 ,:) ,2); 0];
std_1 = [std(x1_k_normalized(1:2 ,:) ,0 ,2); 0]; 
std_2 = [std(x2_k_normalized(1:2 ,:) ,0 ,2); 0]; 
N1 = [ 1/std_1(1)   , 0             , -mean_1(1)/std_1(1); 
       0            , 1/std_1(2)    , -mean_1(2)/std_1(2);
       0            , 0             , 1 ];
N2 = [ 1/std_2(1)   , 0             , -mean_2(1)/std_2(1); 
       0            , 1/std_2(2)    , -mean_2(2)/std_2(2);
       0            , 0             , 1 ];
x1n = N1 * x1_k_normalized; 
x2n = N2 * x2_k_normalized; 
% Normalize the matrices
P1n = N1 * P1; 
P2_1n = N2 * P2_1; 
P2_2n = N2 * P2_2; 
P2_3n = N2 * P2_3; 
P2_4n = N2 * P2_4; 
% Triangulate the points using DLT for each of the four camera solutions
X_3D_1 = triangulate(P1n, P2_1n, x1n, x2n);
X_3D_2 = triangulate(P1n, P2_2n, x1n, x2n);
X_3D_3 = triangulate(P1n, P2_3n, x1n, x2n);
X_3D_4 = triangulate(P1n, P2_4n, x1n, x2n);
% Determine for which of the solutions the points are in front fo the cameras
n = size(x1n, 2);
in_front_counter = [0 0 0 0];
for i = 1:n
    % Project the 3D point to the 2 cameras
    x1_1 = P1 * X_3D_1(:, i);
    x1_2 = P1 * X_3D_2(:, i);
    x1_3 = P1 * X_3D_3(:, i);
    x1_4 = P1 * X_3D_4(:, i);
    x2_1 = P2_1 * X_3D_1(:, i);
    x2_2 = P2_2 * X_3D_2(:, i);
    x2_3 = P2_3 * X_3D_3(:, i);
    x2_4 = P2_4 * X_3D_4(:, i);
    % Add to the counter if the point is in front of both cameras
    if (x1_1(3) >= 0 & x2_1(3) >= 0)
        in_front_counter(1) = in_front_counter(1) + 1;
    end
    if (x1_2(3) >= 0 & x2_2(3) >= 0)
        in_front_counter(2) = in_front_counter(2) + 1;
    end
    if (x1_3(3) >= 0 & x2_3(3) >= 0)
        in_front_counter(3) = in_front_counter(3) + 1;
    end
    if (x1_4(3) >= 0 & x2_4(3) >= 0)
        in_front_counter(4) = in_front_counter(4) + 1;
    end
end
% Select the camera and points s.t. they have the highest number of points in front of the cameras
P2_cameras = {P2_1, P2_2, P2_3, P2_4}; 
X_3Ds = {X_3D_1, X_3D_2, X_3D_3, X_3D_4}; 
index_of_max = find(in_front_counter==max(in_front_counter)); 
P2 = P2_cameras{index_of_max}; 
X_3D = X_3Ds{index_of_max}; 

%%
% Compute the corresponding camera matrices for the original (un-normalized) coordinate system
P1_unnormalized = K * P1; 
P2_unnormalized = K * P2; 
x1_proj = pflat(P1_unnormalized * X_3D); 
x2_proj = pflat(P2_unnormalized * X_3D); 
% Plot the image the points and the projected 3D-points in the same figure
figure; 
imagesc(img1); 
hold on; 
plot(x{1}(1,:), x{1}(2,:), 'bo', 'Markersize', 2, 'LineWidth', 2);
plot(x1_proj(1,:), x1_proj(2,:), 'r.', 'Markersize', 2, 'LineWidth', 2);
axis equal;
axis ij; 
figure; 
imagesc(img2); 
hold on; 
plot(x{2}(1,:), x{2}(2,:), 'bo', 'Markersize', 2, 'LineWidth', 2);
plot(x2_proj(1,:), x2_proj(2,:), 'r.', 'Markersize', 2, 'LineWidth', 2);
axis equal;
axis ij; 
% Plot the 3D points and camera centers and principal axes in a 3D plot
C1 = -inv(P1_unnormalized(:, 1:3)) * P1_unnormalized(:, 4);
C1_axis = det(P1_unnormalized(:, 1:3)) * P1_unnormalized(3, 1:3);
C2 = -inv(P2_unnormalized(:, 1:3)) * P2_unnormalized(:, 4);
C2_axis = det(P2_unnormalized(:, 1:3)) * P2_unnormalized(3, 1:3);
figure; 
plot3(X_3D(1,:), X_3D(2,:), X_3D(3,:), 'r.', 'Markersize', 2, 'LineWidth', 2);
hold on; 
plot3(C1(1) ,C1(2), C1(3)  , '.', 'MarkerSize',20);
quiver3(C1(1) ,C1(2) ,C1(3) ,C1_axis(1) ,C1_axis(2) ,C1_axis(3), 1/norm(C1_axis),'linewidth',3);
plot3(C2(1) ,C2(2), C2(3)  , '.', 'MarkerSize',20);
quiver3(C2(1) ,C2(2) ,C2(3) ,C2_axis(1) ,C2_axis(2) ,C2_axis(3), 1/norm(C2_axis),'linewidth',3);
axis equal;
