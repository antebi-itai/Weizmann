% Multiple View Geometry 2021 semester B
% 
% Student: Itai Antebi, 204817498

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Solution for exercise #2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% question 2

% The first camera
[K_1, R_1] = rq(P{1});
K_1 = K_1 ./ K_1(3,3); 
% The first camera after projective transformations 
[K_1_modified_by_T1, R_1_T1] = rq(modified_P_by_T1{1});
K_1_modified_by_T1 = K_1_modified_by_T1 ./ K_1_modified_by_T1(3,3); 
[K_1_modified_by_T2, R_1_T2] = rq(modified_P_by_T2{1});
K_1_modified_by_T2 = K_1_modified_by_T2 ./ K_1_modified_by_T2(3,3); 
% Print rather the transformations are the same
K1_first_and_after_T1_same = isequal(K_1, K_1_modified_by_T1); 
K1_first_and_after_T2_same = isequal(K_1, K_1_modified_by_T2); 
disp("Are K1 of the first camera and K1 of the camera with the projective transformation T1 the same?"); 
disp(mat2str(K1_first_and_after_T1_same));
disp("Are K1 of the first camera and K1 of the camera with the projective transformation T2 the same?"); 
disp(mat2str(K1_first_and_after_T2_same));