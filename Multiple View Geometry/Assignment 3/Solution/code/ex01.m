% Multiple View Geometry 2021 semester B
% 
% Student: Itai Antebi, 204817498

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Solution for exercise #3
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% question 1

% load the data
load('compEx1data.mat');
img1 = imread('kronan1.JPG');
img2 = imread('kronan2.JPG');

%%
% Compute normalization matrices N1 and N2
mean_1 = [mean(x{1}(1:2 ,:) ,2); 0];
mean_2 = [mean(x{2}(1:2 ,:) ,2); 0];
std_1 = [std(x{1}(1:2 ,:) ,0 ,2); 0]; 
std_2 = [std(x{2}(1:2 ,:) ,0 ,2); 0]; 
N1 = [ 1/std_1(1)   , 0             , -mean_1(1)/std_1(1); 
       0            , 1/std_1(2)    , -mean_1(2)/std_1(2);
       0            , 0             , 1 ];
N2 = [ 1/std_2(1)   , 0             , -mean_2(1)/std_2(1); 
       0            , 1/std_2(2)    , -mean_2(2)/std_2(2);
       0            , 0             , 1 ];
%N1 = eye(3); 
%N2 = eye(3); 
% Normalize the image points of the two images with N1 and N2 respectively.
x1n = N1 * x{1};
x2n = N2 * x{2};

%%
% Set up the matrix M in the eight point algorithm
M = [];
n = size(x1n, 2);
for i=1:n
    xx = x2n(:,i)*x1n(:,i)';
    M(i,:) = xx(:)';
end
% Solve the homogeneous least squares system using SVD
[U, S, V] = svd(M);
v = V(:,end);
% Check that the minimum singular value and |Mv| are both small
disp(['The smallest singular value of M is: ', num2str(S(9, 9))]); 
disp(['Norm of M*v is: ', num2str(norm(M*v))])

%%
% Construct the normalized fundamental matrix from the solution v
Fn = reshape(v, [3 3]);
% Make sure that det(Fn) = 0
[U, S, V] = svd(Fn);
S(3,3) = 0; 
Fn = U*S*V'; 
% Check that the epipolar constraints x2n^T Fn x1n = 0 are roughly fulfilled.
figure;
plot(diag(x2n' * Fn * x1n)); 

%%
% Compute the un-normalized fundamental matrix F
F = N2'*Fn*N1; 
F = F / F(3,3);
disp('F is: ');
disp(F); 
% Compute the epipolar lines l = Fx1
l = F * x{1};
l = l./sqrt(repmat(l(1,:).^2 + l(2,:).^2, [3,1])); 
% Pick 20 points in the second image at random and plot these in the same figure as the image.
idx = randsample(n, 20); 
x2_chosen_points = x{2}(:, idx); 
l1_epipolar_lines = l(:, idx); 
figure; 
imagesc(img2); 
hold on; 
plot(x2_chosen_points(1,:), x2_chosen_points(2,:), 'o', 'Markersize', 2, 'LineWidth', 2);
rital(l1_epipolar_lines); 
axis equal;
axis ij; 

%%
% Compute the distance between all the points and their corresponding epipolar lines
dists = abs(sum(l.*x{2})); 
% Plot these in a histogram with 100 bins.
figure;
hist(dists, 100); 
% What is the mean distance?
disp(['The mean epipolar distances with normalization is: ', num2str(mean(dists))]); 
disp(['The mean epipolar distances without normalization is: ', '0.48784']); 
