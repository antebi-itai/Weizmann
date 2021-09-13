% Multiple View Geometry 2021 semester B
% 
% Student: Itai Antebi, 204817498

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Solution for exercise #3
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% question 2
% compute the camera matrices in Exercise 4
P1 = [eye(3), zeros(3,1)]; 
e2 = null(F'); 
e2x = [0 -e2(3) e2(2); e2(3) 0 -e2(1); -e2(2) e2(1) 0];
P2 = [e2x*F, e2]; 
P1n = N1 * P1; 
P2n = N2 * P2; 
% Use triangulation (with DLT) to compute the 3D-points.
n = size(x1n, 2);
X_3D = zeros(4, n);
lambdas = zeros(2, n);
for i = 1:n
    x1_i = x1n(:, i); 
    x2_i = x2n(:, i); 
    M_i = zeros(6, 6);
    M_i(1:3, 1:4) = P1n;
    M_i(4:6, 1:4) = P2n;
    M_i(1:3,5) = -x1_i;
    M_i(4:6,6) = -x2_i;
    [Ui, Si, Vi] = svd(M_i);
    Xi = pflat(Vi(1:4, 6));
    lambda_i = Vi(5:6, 6);
    X_3D(:, i) = Xi;
    lambdas(:, i) = lambda_i; 
end
% Project the computed points back into the two images
proj_P1 = pflat(P1 * X_3D); 
proj_P2 = pflat(P2 * X_3D); 
% For the two images, Plot both the image, the image points, and the projected 3D points in the same figure.
figure; 
imagesc(img1); 
hold on; 
plot(x{1}(1,:), x{1}(2,:), 'bo', 'Markersize', 2, 'LineWidth', 2);
plot(proj_P1(1,:), proj_P1(2,:), 'r.', 'Markersize', 2, 'LineWidth', 2);
axis equal;
axis ij; 
figure; 
imagesc(img2); 
hold on; 
plot(x{2}(1,:), x{2}(2,:), 'bo', 'Markersize', 2, 'LineWidth', 2);
plot(proj_P2(1,:), proj_P2(2,:), 'r.', 'Markersize', 2, 'LineWidth', 2);
axis equal;
axis ij; 
% Plot the 3D-points in a 3D plot
figure; 
plot3(X_3D(1,:), X_3D(2,:), X_3D(3,:), 'r.', 'Markersize', 2, 'LineWidth', 2);
axis equal;
