%% 
% Final HW solution 
% Joseph Georgeson  336178264
% Itai Antebi       204817498
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% In the final assignment we analyze voting data in Israel, making
% correlations between cities and party voting trends. We go on to compare
% this pattern between elections.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
clc;
clear;
close all;
%% Q1
% 1. Load the data: read the data in the file ‘Kneset_result_2020a.xlsx'
% to a table.
disp('*** Question 1 ***');

full_table_data = readtable('Kneset_result_2020a.xlsx');
NUM_OF_HEADER_COLUMNS = 6; 

disp('  ');

%% Q2
disp('*** Question 2 ***');

NUM_OF_PARTIES_TO_EXPLODE = 5; 
THRESHOLD_PERCENT = 3.25; 

total_votes_per_party = sum(full_table_data{:,NUM_OF_HEADER_COLUMNS+1:end}); % this is only valid votes

sum_votes_invalid = sum(full_table_data.votes_invalid(:));
sum_votes_valid = sum(full_table_data.votes_valid(:));
sum_votes_total = sum(full_table_data.votes_total(:));
sum_registered_voters = sum(full_table_data.registered(:));
threshold_number = sum_votes_valid * THRESHOLD_PERCENT / 100;

figure;
subplot(2,2,1);
col_names = string(full_table_data.Properties.VariableNames);
col_names_subset = col_names(NUM_OF_HEADER_COLUMNS+1:end);
bar(total_votes_per_party);
set(gca, 'xtick', 1:size(col_names_subset, 2), 'xticklabel', col_names_subset); 
yline(threshold_number);
title('Total counts per party');
ylabel('Vote counts');
xlim=get(gca,'xlim');
text(xlim(2),threshold_number,'\leftarrow vote threshold');

subplot(2,2,2);
bar(log(total_votes_per_party));
set(gca, 'xtick', 1:size(col_names_subset, 2), 'xticklabel', col_names_subset); 
yline(log(threshold_number));
title('log(Total counts per party)');
ylabel('log(Vote counts)');
xlim=get(gca,'xlim');
text(xlim(2),log(threshold_number),'\leftarrow vote threshold');

subplot(2,2,3);
[sorted_votes,idx] = sort(total_votes_per_party,'descend');
col_names_idx = col_names_subset(idx);
num_of_parties = size(col_names_subset, 2); 
str = strings(1,num_of_parties);
str(1:NUM_OF_PARTIES_TO_EXPLODE) = col_names_idx(1:NUM_OF_PARTIES_TO_EXPLODE);
labels = str;
explode = [ones(1,NUM_OF_PARTIES_TO_EXPLODE) zeros(1, num_of_parties - NUM_OF_PARTIES_TO_EXPLODE)];
pie(sorted_votes, explode, labels);
title('Total counts per party');

subplot(2,2,4);
set(gca,'visible','off');
set(gca,'xtick',[]);
text(0,1, sprintf("1. Total registered voters: %d",sum_registered_voters));
text(0,.9, sprintf("2. Total voters: %d",sum_votes_total));
text(0,.8, sprintf("3. Total voting rate (in percentage): %g",round(100*sum_votes_total/sum_registered_voters,2)));
text(0,.7, sprintf("4a. Total valid votes: %d",sum_votes_valid));
text(0,.6, sprintf("4b. Total invalid votes: %d",sum_votes_invalid));
text(0,.5, "5a. Votes threshold (percentage): 3.25%");
text(0,.4, sprintf("5b. Votes threshold (counts): %g",threshold_number));

valid_votes = full_table_data.votes_valid;
total_votes = full_table_data.votes_total;
perc_valid = valid_votes./total_votes;

[valid_votes_sort,idx_valid] = sort(perc_valid,'descend');
settlement_names = full_table_data.settlement_name;
names_sort = settlement_names(idx_valid);
disp("406 settlements had no invalid votes...");
disp("Top 10 settlements with valid votes:");
names_sort(1:10)

disp("Bottom 10 settlements with valid votes:");
flip(names_sort(end-9:end))

disp('  ');

%% Q3
%3.a
disp('*** Question 3 ***');

settlement_votes = full_table_data{:,NUM_OF_HEADER_COLUMNS+1:end}; % this is only valid votes
total_votes_per_party = sum(full_table_data{:,NUM_OF_HEADER_COLUMNS+1:end});
[rho,pval] = corr(settlement_votes.',total_votes_per_party.');

%3.b
[rho_sort,rho_idx] = sort(rho,'descend');

settlement_names = full_table_data.settlement_name;
names_rho_sort = settlement_names(rho_idx);
disp("Top 10 settlements with highest correlation:");
names_rho_sort(1:10)
disp("Bottom 10 settlements with lowest correlation:");
flip(names_rho_sort(end-9:end))

%3.c
[all_rho,all_pval] = corr(settlement_votes.');

%3.d
all_rho_flat = reshape(all_rho.',1,[]);

[sort_all_rho, idx_all_rho] = sort(all_rho_flat, 'descend');

sort_all_idx_unflat = reshape(idx_all_rho, size(all_rho,1), size(all_rho,2));
[top_row,top_col] = find(sort_all_idx_unflat==(size(all_rho, 1) + 1));
[low_row,low_col] = find(sort_all_idx_unflat==size(all_rho_flat,2));

top_pair = append("The settlements with highest correlation are : ", ...
    string(settlement_names(top_row)), " and ", string(settlement_names(top_col)));
low_pair = append("The settlements with lowest correlation are : ", ...
    string(settlement_names(low_row)), " and ", string(settlement_names(low_col)));
disp(top_pair);
disp(low_pair);

disp('  ');

%% Q4
%4.a
disp('*** Question 4 ***');

sample = full_table_data.settlement_name;
features = full_table_data.Properties.VariableNames;
sample_data = full_table_data{:,NUM_OF_HEADER_COLUMNS+1:end}; % this is only valid votes
disp(append("There are ", string(size(sample,1)), " samples (settlements)."));
disp(append("There are ", string(size(sample_data,2)), " features (different parties)."));

%4.b
rng(0); % For reproducibility
num_clusters = 5;
idx_init = kmeans(sample_data,num_clusters);
figure; 
[silh,h] = silhouette(sample_data,idx_init);
xlabel('Silhouette Value');
ylabel('Cluster');

%4.c
rng(0); % For reproducibility
idx1 = kmeans(sample_data,num_clusters,'Distance','sqeuclidean','Display','final');
rng(0); % For reproducibility
idx2 = kmeans(sample_data,num_clusters,'Distance','cosine','Display','final');
rng(0); % For reproducibility
idx3 = kmeans(sample_data,num_clusters,'Distance','correlation','Display','final');

figure;
subplot(3,1,1);
[silh1,h1] = silhouette(sample_data,idx1,'sqeuclidean');
xlabel('Silhouette Value');
ylabel('Cluster (sqeuclidean)');

subplot(3,1,2);
[silh2,h2] = silhouette(sample_data,idx2,'cosine');
xlabel('Silhouette Value');
ylabel('Cluster (cosine)');

subplot(3,1,3);
[silh3,h3] = silhouette(sample_data,idx3,'correlation');
xlabel('Silhouette Value');
ylabel('Cluster (correlation)');

%4.d
num_reps = 50;

rng(0); % For reproducibility
idx1reps = kmeans(sample_data,num_clusters,'Distance','sqeuclidean','Display','final','Replicates',num_reps);
rng(0); % For reproducibility
idx2reps = kmeans(sample_data,num_clusters,'Distance','cosine','Display','final','Replicates',num_reps);
rng(0); % For reproducibility
idx3reps = kmeans(sample_data,num_clusters,'Distance','correlation','Display','final','Replicates',num_reps);

figure;
subplot(3,1,1);
[silh1,h1] = silhouette(sample_data,idx1reps,'sqeuclidean');
xlabel('Silhouette Value');
ylabel('Cluster (sqeuclidean w/reps)');

subplot(3,1,2);
[silh2,h2] = silhouette(sample_data,idx2reps,'cosine');
xlabel('Silhouette Value');
ylabel('Cluster (cosine w/reps)');

subplot(3,1,3);
[silh3,h3] = silhouette(sample_data,idx3reps,'correlation');
xlabel('Silhouette Value');
ylabel('Cluster (correlation w/reps)');

%4.e

figure;
for k = 2:10
    rng(0);
    idx_now = kmeans(sample_data,k);
    subplot(3,3,(k-1));
    [silh,h] = silhouette(sample_data,idx_now);
    ylabel(sprintf('Cluster k=%d', k));
end

%4.f
k_range = 2:10;

for metric = ["sqeuclidean", "cosine", "correlation"]
    
    figure;    
    for k = k_range
        
        rng(0);
        idx_now = kmeans(sample_data,k,'Distance',metric);
        subplot(3,4,(k-1));
        [silh,h] = silhouette(sample_data,idx_now);
        xlabel(append('Silhouette Value: ',metric));
        ylabel(sprintf('Cluster k=%d', k));
        
    end
    
    rng(0);
    E = evalclusters(sample_data,'kmeans','silhouette','klist',k_range,'Distance',metric);
    % save for Q5
    if metric == "correlation"
        E_correlation = E;
    end
    subplot(3,4,10);
    plot(E);
    hold on;
    plot(E.OptimalK, E.CriterionValues(E.OptimalK),'r-*');
    text((E.OptimalK + 0.5), E.CriterionValues(E.OptimalK), append("Optimal k for metric ", metric," is: ",string(E.OptimalK)))
end

%4.g Can you explain why using the 'sqeuclidean' created different results? 

% sqeuclidean takes into account the weight and magnitude between vectors

%How can you fix it?

% normalize the vectors before calculating the distance 

disp('  ');

%% Q5

disp('*** Question 5 ***');

rng(0);
idx_optimal_correlation = kmeans(sample_data, E_correlation.OptimalK, 'Distance', 'correlation');

disp('  ');

%% Q6
disp('*** Question 6 ***');

figure; 

%6.b
clusters = 1:E_correlation.OptimalK; 
counts = histc(idx_optimal_correlation, clusters);
percentages = (counts / sum(counts)) * 100; 
labels = cellstr(num2str(clusters(:))); 
colors = zeros(size(clusters, 2), 3);
for cluster = clusters
    colors(cluster, :) = [rand, rand, rand]; 
    labels{cluster} = "Cluster #" + labels{cluster} + " - " + num2str(percentages(cluster)) + "%";
end
subplot(2,2,1);
ax = gca(); 
h = pie(ax, percentages, labels); 
ax.Colormap = colors;

%6.c
subplot(2,2,2);
for cluster = clusters
    cluster_data = full_table_data(idx_optimal_correlation==cluster, NUM_OF_HEADER_COLUMNS+1:end); 
    cluster_votes = sum(cluster_data{:,:}, 1); 
    cluster_pattern = (cluster_votes / sum(cluster_votes)) * 100; 
    stem(cluster_pattern, 'Color', colors(cluster, :)); 
    hold on; 
end
general_data = full_table_data(:, NUM_OF_HEADER_COLUMNS+1:end); 
general_votes = sum(general_data{:,:}, 1); 
general_pattern = (general_votes / sum(general_votes)) * 100; 
plot(general_pattern, 'k', 'LineWidth', 3); 
% legends and labels
labels = cellstr(num2str(clusters(:))); 
for cluster = clusters
    labels{cluster} = "Cluster #" + labels{cluster};
end
labels{end + 1} = "General Public"; 
legend(labels); 
xlabel("Parties");
col_names = string(full_table_data.Properties.VariableNames);
col_names_subset = col_names(NUM_OF_HEADER_COLUMNS+1:end);
set(gca, 'xtick', 1:size(col_names_subset, 2), 'xticklabel', col_names_subset); 
ylabel("Party Voting Percentage");
hold off; 

%6.d
subplot(2,2,3);
for cluster = clusters
    cluster_data = full_table_data(idx_optimal_correlation==cluster, NUM_OF_HEADER_COLUMNS+1:end); 
    cluster_votes = sum(cluster_data{:,:}, 1); 
    [rho,pval] = corr(cluster_votes.',general_data{:,:}.');
    histogram(rho, 'FaceColor', colors(cluster, :))
    hold on;
end
% legends and labels
legend(labels(1:end-1)); 
xlabel("correlation values");
ylabel("number of settlements the cluster correlates to");
hold off; 

%6.e
subplot(2,2,4);
num_of_votes = sum(general_data{:,:}, 2); 
voting_rates = (full_table_data.('votes_valid') ./ full_table_data.('registered')) * 100; 
[rho,pval] = corr(general_data{:,:}.', general_votes.');
plot_colors = colors(idx_optimal_correlation, :); 
for cluster = clusters
    cluster_mask = idx_optimal_correlation==cluster; 
    plot3(num_of_votes(cluster_mask), voting_rates(cluster_mask), rho(cluster_mask), 'marker','.','markeredgecolor', colors(cluster, :), 'LineStyle', 'none'); 
    hold on; 
end
% legends and labels
xlabel("num of votes");
ylabel("voting rates");
zlabel("correlation to general population"); 
view(52.3793, 35.4026); 
hold off; 

%6.f Select two clusters and try to explain their results using the figure
% you created. What is different between those groups?

% Cluster 3 - vast majority voted for "Hareshima Hameshutefet", consist of
% about 10% of the population and are highly uncorrelated with the general public
% Cluster 2 - almost 50% voted for "Kachol Lavan", consist of about 50% of
% the population and are much more correlated with the general public

disp('  ');

%% Q7
disp('*** Question 7 ***');

disp('  ');

%% Q8
disp('*** Question 8 ***');

% a. Load the data from all three elections
elections = {"2019a", "2019b", "2020a"}; 
full_table_data_2019a = readtable('Kneset_result_2019a.xlsx');
full_table_data_2019b = readtable('Kneset_result_2019b.xlsx');
full_table_data_2020a = readtable('Kneset_result_2020a.xlsx');

% b. Find which parties participated in all three elections 
parties_2019a = full_table_data_2019a.Properties.VariableNames(NUM_OF_HEADER_COLUMNS+1:end);
parties_2019b = full_table_data_2019b.Properties.VariableNames(NUM_OF_HEADER_COLUMNS+1:end);
parties_2020a = full_table_data_2020a.Properties.VariableNames(NUM_OF_HEADER_COLUMNS+1:end);
parties_in_all_elections = intersect(parties_2019a, intersect(parties_2019b, parties_2020a)); 
fprintf("Parties that participated in all three elections: \n"); 
fprintf('%s\n', parties_in_all_elections{:});

% c. Create scatter plots of votes in between all pairs of elections.
num_votes_per_relevant_party_2019a = sum(table2array(full_table_data_2019a(:, parties_in_all_elections)), 1); 
num_votes_per_relevant_party_2019b = sum(table2array(full_table_data_2019b(:, parties_in_all_elections)), 1); 
num_votes_per_relevant_party_2020a = sum(table2array(full_table_data_2020a(:, parties_in_all_elections)), 1); 
num_votes_per_relevant_party = [num_votes_per_relevant_party_2019a; ...
                                num_votes_per_relevant_party_2019b; ...
                                num_votes_per_relevant_party_2020a]';
num_of_parties = size(num_votes_per_relevant_party, 1); 
num_of_elections = size(num_votes_per_relevant_party, 2); 

figure(); 
hold on; 
colors = zeros(num_of_parties, 3);
for party_idx = 1:num_of_parties
    colors(party_idx, :) = [rand, rand, rand]; 
    x = 1:num_of_elections; 
    y = num_votes_per_relevant_party(party_idx,:); 
    scatter(x, y, [], colors(party_idx, :));
    % e. Fit a regression line and add it to the figures above.
    c = polyfit(x,y,1);
    y_est = polyval(c,x);
    plot(x,y_est,'--','LineWidth',2,'Color',colors(party_idx, :));
end
% add legend and ticks
parties_in_all_elections_legend = cell(1, 2*num_of_parties);
for idx = 1:num_of_parties
    parties_in_all_elections_legend{2*idx-1} = parties_in_all_elections{idx}; 
    parties_in_all_elections_legend{2*idx} = strcat(parties_in_all_elections{idx}, ' - linear fit'); 
end
legend(parties_in_all_elections_legend ,'Location', 'NorthEastOutside');
set(gca, 'xtick', [1:num_of_elections],'xticklabel',elections)
ylabel("number of votes"); 

% d. Compute the correlation between the elections and print it in the title of the graphs above.
[rho,pval] = corr(num_votes_per_relevant_party, num_votes_per_relevant_party);
first_second_corr_text = sprintf("Correlation between first & second elections - %.4f", rho(1, 2)); 
second_third_corr_text = sprintf("Correlation between second & third elections - %.4f", rho(2, 3)); 
title({first_second_corr_text, second_third_corr_text}); 

% f. Create a bar graph of all three elections
figure(); 
bar(num_votes_per_relevant_party);
set(gca, 'xtick', 1:num_of_parties, 'xticklabel', parties_in_all_elections); 
yline(threshold_number);
title('Total counts per party');
legend(elections); 

disp('  ');

%% Q9
disp('*** Question 9 ***');

% We ask the question 'Which parties participated in all 3 elections and
% passed the threshold, and which had the highest growth rate?'

% Find which parties participated in all three elections & entered the passed the threshold
participated_and_passed_2019a = parties_2019a(sum(table2array(full_table_data_2019a(:, NUM_OF_HEADER_COLUMNS+1:end)), 1) > threshold_number); 
participated_and_passed_2019b = parties_2019b(sum(table2array(full_table_data_2019b(:, NUM_OF_HEADER_COLUMNS+1:end)), 1) > threshold_number); 
participated_and_passed_2020a = parties_2020a(sum(table2array(full_table_data_2020a(:, NUM_OF_HEADER_COLUMNS+1:end)), 1) > threshold_number); 
parties_in_all_elections_and_passed = intersect(participated_and_passed_2019a, intersect(participated_and_passed_2019b, participated_and_passed_2020a)); 
fprintf("Parties that participated in all three elections and passed them: \n"); 
fprintf('%s\n', parties_in_all_elections_and_passed{:});
% Find which of them increased her vote count by the highest ration
num_votes_per_relevant_party_2019a = sum(table2array(full_table_data_2019a(:, parties_in_all_elections_and_passed)), 1); 
num_votes_per_relevant_party_2020a = sum(table2array(full_table_data_2020a(:, parties_in_all_elections_and_passed)), 1); 
[argvalue, argmax] = max(num_votes_per_relevant_party_2020a ./ num_votes_per_relevant_party_2019a);
best_growth_str = strcat('Party', {' '}, parties_in_all_elections_and_passed{argmax}, {' '}, ...
                         'had the highest growth rate between 2019a and 2020a - ', {' '}, ...
                         num2str(round(argvalue * 100, 0)), '%'); 
disp('  ');
disp(best_growth_str{:}); 

disp('  ');

