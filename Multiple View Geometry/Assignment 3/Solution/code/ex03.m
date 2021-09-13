% Multiple View Geometry 2021 semester B
% 
% Student: Itai Antebi, 204817498

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Solution for exercise #3
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% question 3
% load the data
load('compEx3data.mat'); 

%%
% Normalize the image points using the inverse of K.
x1_k_normalized = inv(K) * x{1}; 
x2_k_normalized = inv(K) * x{2}; 
% Set up the matrix M in the eight point algorithm
M = [];
n = size(x1_k_normalized, 2);
for i=1:n
    xx = x2_k_normalized(:,i)*x1_k_normalized(:,i)';
    M(i,:) = xx(:)';
end
% Solve the homogeneous least squares system using SVD
[U, S, V] = svd(M);
v = V(:,end);
% Check that the minimum singular value and |Mv| are both small
disp(['The smallest singular value of M is: ', num2str(S(9, 9))]); 
disp(['Norm of M*v is: ', num2str(norm(M*v))])

%%
% Construct the Essential matrix from the solution v
Eapprox = reshape(v, [3 3]);
% Don’t forget to make sure that E has two equal singular values and the third one zero
[U, S, V] = svd(Eapprox);
if det(U*V')>0
    E = U*diag([1 1 0])*V';
else
    V = -V;
    E = U*diag([1 1 0])*V';
end
disp('E is: ');
disp(E / E(3,3)); 
% Check that the epipolar constraints x2n^T E x1n = 0 are roughly fulfilled.
figure;
plot(diag(x2_k_normalized' * E * x1_k_normalized)); 

%%
% Compute the fundamental matrix for the un-normalized coordinate system from the essential matrix
F = (inv(K))'*E*inv(K); 

%%
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
disp(['The mean epipolar distances is: ', num2str(mean(dists))]); 
