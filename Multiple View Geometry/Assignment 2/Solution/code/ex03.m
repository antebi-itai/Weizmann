% Multiple View Geometry 2021 semester B
% 
% Student: Itai Antebi, 204817498

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Solution for exercise #2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% question 3

% Load data
load('compEx3data.mat');

% Normalize the measured points
mean_1 = mean(x{1}, 2);
mean_1(3) = 0;
std_1 = std(x{1}');
std_1 = [std_1(1:2), 1]';
N1 = [ 1/std_1(1)   , 0             , -mean_1(1)/std_1(1); 
       0            , 1/std_1(2)    , -mean_1(2)/std_1(2);
       0            , 0             , 1 ];
% N1 = eye(3); 
% This is done using matmul for later convenience
%Nx1 = (x{1} - mean_1) ./ std_1; 
Nx1 = N1 * x{1};

mean_2 = mean(x{2}, 2);
mean_2(3) = 0;
std_2 = std(x{2}');
std_2 = [std_2(1:2), 1]';
N2 = [ 1/std_2(1)   , 0             , -mean_2(1)/std_2(1); 
       0            , 1/std_2(2)    , -mean_2(2)/std_2(2);
       0            , 0             , 1 ];
% N2 = eye(3); 
% This is done using matmul for later convenience
%Nx2 = (x{2} - mean_2) ./ std_2; 
Nx2 = N2 * x{2};

% Plot the normalized points
figure;
plot(Nx1(1, :), Nx1(2, :), '.');
axis equal;
figure;
plot(Nx2(1, :), Nx2(2, :), '.');
axis equal;
% It looks like the points are centered around (0, 0) with mean distance 1 to (0, 0)

%% a. 
% Set up the DLT equations for resectioning
n = size(Xmodel, 2); 
hom_X = [Xmodel; ones(1,n)]';

M1 = zeros(3 * n, 12 + n);
M2 = zeros(3 * n, 12 + n);

for i=1:n
    M1((i-1)*3+1, 1:4) = hom_X(i, :);
    M1((i-1)*3+2, 5:8) = hom_X(i, :);
    M1((i-1)*3+3, 9:12) = hom_X(i, :);
    M1((i-1)*3+1, 12+i) = -Nx1(1,i);
    M1((i-1)*3+2, 12+i) = -Nx1(2,i);
    M1((i-1)*3+3, 12+i) = -1;
    
    M2((i-1)*3+1, 1:4) = hom_X(i, :);
    M2((i-1)*3+2, 5:8) = hom_X(i, :);
    M2((i-1)*3+3, 9:12) = hom_X(i, :);
    M2((i-1)*3+1, 12+i) = -Nx2(1,i);
    M2((i-1)*3+2, 12+i) = -Nx2(2,i);
    M2((i-1)*3+3, 12+i) = -1;
end

[U1, Sigma_1, V1] = svd(M1);
[U2, Sigma_2, V2] = svd(M2);
v1 = V1(:,end);
v2 = V2(:,end);

disp(['The smallest singular value of M1 is: ', num2str(Sigma_1(12 + n, 12 + n))]); 
disp(['The smallest singular value of M2 is: ', num2str(Sigma_2(12 + n, 12 + n))]); 
disp(['Norm of M1*v1 is: ', num2str(norm(M1*v1))])
disp(['Norm of M2*v2 is: ', num2str(norm(M2*v2))])
% The smallest singular values in both cases is at e-2 scale - very close to 0.
% The norm of Mv in both cases is at e-2 scale - very close to 0.

% Extract the entries of the camera from the solution and set up the camera matrix.
N_P1 = reshape(v1(1:12), [4, 3])'; 
N_P2 = reshape(v2(1:12), [4, 3])'; 
% Make sure that you select the solution where the points are in front of the camera
N_proj_X_P1 = N_P1 * hom_X';
N_proj_X_P2 = N_P2 * hom_X';
if N_proj_X_P1(3, 1) < 0
    N_P1 = -N_P1;
end
if N_proj_X_P2(3, 1) < 0
    N_P2 = -N_P2;
end

%% b. 
proj_X_P1 = N1 \ pflat(N_proj_X_P1);
proj_X_P2 = N2 \ pflat(N_proj_X_P2);

% Plot the measured image points in the same figure.
figure;
hold on
plot(proj_X_P1(1 , :), proj_X_P1(2 , :) , 'b*');
plot(x{1}(1 , :), x{1}(2 , :) , 'ro');
axis equal;
figure;
hold on
plot(proj_X_P2(1 , :), proj_X_P2(2 , :) , 'b*');
plot(x{2}(1 , :), x{2}(2 , :) , 'ro');
axis equal;
% Yes, they are indeed close to each other

% Plot the camera centers and viewing directions in the same plot as the 3D model points.
P1 = N1 \ N_P1; 
P2 = N2 \ N_P2; 
C1 = -(P1(:,1:3))'*P1(:,4);
C2 = -(P2(:,1:3))'*P2(:,4);
d1 = det(P1(:,1:3))*P1(3,1:3);
d2 = det(P2(:,1:3))*P2(3,1:3);

cameras = {P1, P2};
figure;
plotcams(cameras);
hold on;
plot3(hom_X(:, 1), hom_X(:, 2), hom_X(:, 3), 'bo');
axis equal;
view(-20,5)
% Yes, the result look reasonable (though flipped :) )

% Compute the inner parameters of the first camera using rq.m.
[K1, R1] = rq(P1);
[K2, R2] = rq(P2);
% We can know that these are the "true" parameters since we are given both
% the precise point model and the measured projections x. 
% The ambiguity in Exercise 1 came from the fact that we didn't know the
% point model (3D-points), we only knew the measured projections. Thus, we
% could alter the 3D-points and the cameras and reach a different, yet
% still consistent, result. 

% Compute the RMS error
N_rmse_1 = sqrt(mean(sum((x{1} - proj_X_P1).^ 2), 'all')); 
disp(['The RMSE for camera 1 WITH normalization is: ', num2str(N_rmse_1)]); 
disp(['The RMSE for camera 1 WITHOUT normalization is: 4.9026']); 
N_rmse_2 = sqrt(mean(sum((x{2} - proj_X_P2).^ 2), 'all')); 
disp(['The RMSE for camera 2 WITH normalization is: ', num2str(N_rmse_2)]); 
disp(['The RMSE for camera 2 WITHOUT normalization is: 3.8968']); 
