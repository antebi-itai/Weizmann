%% 
% HW_2 solution
% Joseph Georgeson  336178264
% Itai Antebi       204817498

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% In HW_2 below, we are given a dataset to practice matrix manipulations
% to become familiar with 'sort' and indexing to subset data as requested
% in the HW prompt.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;
clc;

%% Q1
doc sort
% Input     - 
%   required:   A - a vector / matrix / multidimentional array
%   optional:   dim - the dimesion to sort
%               direction - 'ascend' or 'descend'
%               Name,Value - use a different comparison method
% 
% Output    - 
%   The same array after sorting the 'dim' dimension in the relevant
%   direction / method. 
% 
% If we expect the function to produce two outputs, then the second output
% would be the indices of the sorted array in the input array. 

%% Q2
% load the data into the workspace
load('data_hw2_2021b.mat'); 

%% Q3
% create a matrix with all four tests together concatenated
tests = [test1_scores , test2_scores , test3_scores , test4_scores];

%% Q4
% find the median, mean, max and minimum for the 4 tests
tests_med   = median(tests); 
tests_mean  = mean(tests);
tests_max   = max(tests);
tests_min   = min(tests); 
% display the results
fprintf(['Test No.1  results: \n', ...
    'median: ', num2str(tests_med(1)),  '\t\t', ...
    'mean: ',   num2str(tests_mean(1)), '\t\t', ...
    'max: ',    num2str(tests_max(1)),  '\t', ...
    'min: ',    num2str(tests_min(1)), '\n']);
fprintf(['Test No.2  results: \n', ...
    'median: ', num2str(tests_med(2)),  '\t\t', ...
    'mean: ',   num2str(tests_mean(2)), '\t\t', ...
    'max: ',    num2str(tests_max(2)),  '\t\t', ...
    'min: ',    num2str(tests_min(2)), '\n']);
fprintf(['Test No.3  results: \n', ...
    'median: ', num2str(tests_med(3)),  '\t\t', ...
    'mean: ',   num2str(tests_mean(3)), '\t\t', ...
    'max: ',    num2str(tests_max(3)),  '\t\t', ...
    'min: ',    num2str(tests_min(3)), '\n']);
fprintf(['Test No.4  results: \n', ...
    'median: ', num2str(tests_med(4)),  '\t\t', ...
    'mean: ',   num2str(tests_mean(4)), '\t\t', ...
    'max: ',    num2str(tests_max(4)),  '\t\t', ...
    'min: ',    num2str(tests_min(4)), '\n\n']);

%% Q5
% find highest, lowest and average of the medians
tests_med_sorted    = sort(tests_med); 
highest_test_med    = tests_med_sorted(end); 
lowest_test_med     = tests_med_sorted(1); 
avg_test_med        = mean(tests_med_sorted); 
% display the results rounded down
fprintf(['The lowest median score is:',             '\t\t',     num2str(floor(lowest_test_med)),    '\n']);
fprintf(['The highest median score is:',            '\t\t',     num2str(floor(highest_test_med)),   '\n']);
fprintf(['The average score of all medians is:',    '\t',     num2str(floor(avg_test_med)),       '\n\n']);

%% Q6
% find the participants weighted average of the 4 tests
participants_weighted_average = sum(tests .* [0.20, 0.15, 0.35, 0.30], 2); 
% check the affect of the experiment result
participants_weighted_average_sorted = sort(participants_weighted_average); 
lowest_weighted_score = participants_weighted_average_sorted(1); 
heighest_weighted_score = participants_weighted_average_sorted(end); 
avg_weighted_score = mean(participants_weighted_average_sorted);

% How do the weights affect the experiment results?
% 
% Weighting the tests changes the lowest score by -20 points, the highest
% score by +3 points, and the average score -3 points.

% print the results (rounded up)
fprintf(['The lowest weighted score is:',   '\t\t', num2str(ceil(lowest_weighted_score)),   '\n']);
fprintf(['The highest weighted score is:',  '\t\t', num2str(ceil(heighest_weighted_score)), '\n']);
fprintf(['The average weighted score is:',  '\t\t', num2str(ceil(avg_weighted_score)),      '\n\n']);

%% Q7
% find the 3 participants that got the highest scores in test number 4
[sorted_scores_test4, indices_sorted_scores_test4] = sort(test4_scores, 'descend'); 
top_3_scores_test_4_indices = indices_sorted_scores_test4(1:3); 
top_3_scores_test_4_dates   = Date(top_3_scores_test_4_indices, :); 
top_3_scores_test_4_gender  = Gender(top_3_scores_test_4_indices, :); 
top_3_scores_test_4_scores  = test4_scores(top_3_scores_test_4_indices); 
% display the results in descending oreder
fprintf(['1. Participant no. ', ...
    num2str(top_3_scores_test_4_indices(1)),    '\t', ...
    top_3_scores_test_4_dates(1, :),            '\t', ...
    top_3_scores_test_4_gender(1, :),           '\t', ...
    num2str(top_3_scores_test_4_scores(1)),     '\n']);
fprintf(['2. Participant no. ', ...
    num2str(top_3_scores_test_4_indices(2)),    '\t', ...
    top_3_scores_test_4_dates(2, :),            '\t', ...
    top_3_scores_test_4_gender(2, :),           '\t', ...
    num2str(top_3_scores_test_4_scores(2)),     '\n']);
fprintf(['3. Participant no. ', ...
    num2str(top_3_scores_test_4_indices(3)),    '\t', ...
    top_3_scores_test_4_dates(3, :),            '\t', ...
    top_3_scores_test_4_gender(3, :),           '\t', ...
    num2str(top_3_scores_test_4_scores(3)),     '\n\n']);

%% Q8
% sort the data according to test1 scores in descending order
[sorted_scores_test1, indices_sorted_scores_test1] = sort(test1_scores, 'descend'); 
% find the position of the top 3 from Q7 among the 30 participant’s test1 scores
test4_top1_position_in_test1 = find(indices_sorted_scores_test1 == top_3_scores_test_4_indices(1));
test4_top2_position_in_test1 = find(indices_sorted_scores_test1 == top_3_scores_test_4_indices(2));
test4_top3_position_in_test1 = find(indices_sorted_scores_test1 == top_3_scores_test_4_indices(3));
% display the results
fprintf(['test4 top student #1 is in position: ', num2str(test4_top1_position_in_test1), '\tin test1 scores\n']);
fprintf(['test4 top student #2 is in position: ', num2str(test4_top2_position_in_test1), '\tin test1 scores\n']);
fprintf(['test4 top student #3 is in position: ', num2str(test4_top3_position_in_test1), '\tin test1 scores\n\n']);

%% Q9
% how many females participated in each date
date_1_num_of_female_participants = sum(Gender(1:10)=='f'); 
date_2_num_of_female_participants = sum(Gender(11:20)=='f'); 
date_3_num_of_female_participants = sum(Gender(21:30)=='f'); 
% display the results
fprintf([Date(1,:),     '\t number of females \t', num2str(date_1_num_of_female_participants), '\n']);
fprintf([Date(11,:),    '\t number of females \t', num2str(date_2_num_of_female_participants), '\n']);
fprintf([Date(21,:),    '\t number of females \t', num2str(date_3_num_of_female_participants), '\n\n']);

%% Q10
% remove from the data all people aged over 60 and under 21
YOUNG_AGE   = 21;
OLD_AGE     = 60;
correct_age_indices = find(((Age < OLD_AGE) & (Age >= YOUNG_AGE)) == 1); 
Age = Age(correct_age_indices); 
Date = Date(correct_age_indices, :); 
Gender = Gender(correct_age_indices, :); 
test1_scores = test1_scores(correct_age_indices); 
test2_scores = test2_scores(correct_age_indices); 
test3_scores = test3_scores(correct_age_indices); 
test4_scores = test4_scores(correct_age_indices); 
tests = [test1_scores , test2_scores , test3_scores , test4_scores];
% find again the median, mean, max and minimum for the 4 tests
tests_med   = median(tests); 
tests_mean  = mean(tests);
tests_max   = max(tests);
tests_min   = min(tests); 
% display the results
fprintf(['Test No.1  results: \n', ...
    'median: ', num2str(tests_med(1)),  '\t\t', ...
    'mean: ',   num2str(tests_mean(1)), '\t\t', ...
    'max: ',    num2str(tests_max(1)),  '\t', ...
    'min: ',    num2str(tests_min(1)), '\n']);
fprintf(['Test No.2  results: \n', ...
    'median: ', num2str(tests_med(2)),  '\t\t', ...
    'mean: ',   num2str(tests_mean(2)), '\t\t', ...
    'max: ',    num2str(tests_max(2)),  '\t\t', ...
    'min: ',    num2str(tests_min(2)), '\n']);
fprintf(['Test No.3  results: \n', ...
    'median: ', num2str(tests_med(3)),  '\t\t', ...
    'mean: ',   num2str(tests_mean(3)), '\t\t', ...
    'max: ',    num2str(tests_max(3)),  '\t\t', ...
    'min: ',    num2str(tests_min(3)), '\n']);
fprintf(['Test No.4  results: \n', ...
    'median: ', num2str(tests_med(4)),  '\t\t', ...
    'mean: ',   num2str(tests_mean(4)), '\t\t', ...
    'max: ',    num2str(tests_max(4)),  '\t\t', ...
    'min: ',    num2str(tests_min(4)), '\n\n']);

%% Q11
% find all the places where the score is below 55 and replace it with 60
THRESHOLD = 55; 
MINIMUM_SCORE = 60; 
tests(tests < THRESHOLD) = MINIMUM_SCORE;
% find the new average for each test
tests_new_averages = mean(tests); 
% display the result rounded
fprintf(['Test No.1 new average is:\t', num2str(round(tests_new_averages(1))), '\n']);
fprintf(['Test No.2 new average is:\t', num2str(round(tests_new_averages(2))), '\n']);
fprintf(['Test No.3 new average is:\t', num2str(round(tests_new_averages(3))), '\n']);
fprintf(['Test No.4 new average is:\t', num2str(round(tests_new_averages(4))), '\n\n']);

%% Q12
% select every 3rd participant starting from the first participant
group1 = [1 : 3 : size(Age,1)];
% select every 3rd participant starting from the second participant
group2 = [2 : 3 : size(Age,1)];
% calculate the age median in each group
group1_age_median = median(Age(group1));
group2_age_median = median(Age(group2));
% display the results
fprintf(['Group 1 age median is: \t', num2str(group1_age_median), '\n']);
fprintf(['Group 2 age median is: \t', num2str(group2_age_median), '\n']);
