%% 
% HW_3 solution
% Joseph Georgeson  336178264
% Itai Antebi       204817498

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% In HW_3 below we visualize and subset conductivity data for several
% electrolytes in aqueous solution at 25C
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;
clc;

%% Q1
% a. Load the file conductivity.mat
load('conductivity.mat');

%%
% b. Calculate the mean and the median conductivity of the electrolytes
mean_cond = mean(conductivity ,1);
median_cond = median(conductivity ,1);

%%
% c. Plot the mean and the median conductivity vs. concentration
plot(concentration, mean_cond, 'r-x','LineWidth',2);
hold on;
plot(concentration, median_cond, 'b-x','LineWidth',2);
hold off;
xlabel('Concentration', 'FontSize', 14);
ylabel('Conductivity', 'FontSize', 14);
title('Mean and median conductivity vs concentration', 'FontSize', 16);
legend('Mean','Median', 'Location', 'SouthEastOutside');

%%
% d. Calculate and display in the command window the minimal and maximal
% conductivity values that are shown in the plotted data.
to_exlcude = 'HCl';
margin_buffer = 25;
index2include = find(~contains(electrolyte_names, to_exlcude));

global_min = min(conductivity(index2include,:), [], 'all');
global_max = max(conductivity(index2include,:), [], 'all');
text_min = sprintf('Min conductivity: %.1f', global_min);
text_max = sprintf('Max conductivity: %.1f', global_max);

ylim_min = global_min - margin_buffer;
ylim_max = global_max + margin_buffer;

figure;
plot(concentration, conductivity(index2include,:),'LineWidth', 2);
ylim([ylim_min, ylim_max]);
disp(text_min);
disp(text_max);
title('Conductivity by electrolyte');
xlabel('Concentration', 'FontSize', 14);
ylabel('Conductivity', 'FontSize', 14);
legend(electrolyte_names(index2include), 'Location', 'SouthEastOutside');

%%
% e. Plot the maximal conductivity per each electrolyte  
% (in the given range of concentrations) using the bar function
cond_max_by_electrolyte    = max(conductivity');

figure;
bar(cond_max_by_electrolyte);
set(gca, 'XTick',1:length(electrolyte_names));
set(gca, 'XTickLabel', electrolyte_names);
set(gca, 'XTickLabelRotation', 90);

title('Max conductivity by electrolyte');
xlabel('Electrolyte', 'FontSize', 14);
ylabel('Conductivity', 'FontSize', 14);

%%
% f. Plot in blue lines the conductivity values for molecules have a 
% single chlorine atom. Plot in red lines the conductivity of molecules 
% having two chlorine atoms. Add axis labels a title and a legend. 
% Note: Do not use fixed index for those molecules! make sure to 
% automatically find those molecules using a string comparison 
% function (conatins for example).

Cl_any_str = 'Cl';
Cl2_str = 'Cl2';
Cl_multi_str = 'Cl\d';

Cl_any_index = find(contains(electrolyte_names, Cl_any_str));
Cl2_index = find(contains(electrolyte_names, Cl2_str));
Cl_NOTmulti_index = find(cellfun(@isempty,regexp(electrolyte_names, Cl_multi_str)));
Cl_single_index = intersect(Cl_any_index, Cl_NOTmulti_index);

t_conductivity = conductivity';
Cl1_values = t_conductivity(:, Cl_single_index);
Cl2_values = t_conductivity(:, Cl2_index);

figure
hold on;
plot(concentration, Cl1_values,'b', 'LineWidth', 2);
plot(concentration, Cl2_values,'r', 'LineWidth', 2);
hold off;
legend([electrolyte_names(Cl_single_index), electrolyte_names(Cl2_index)], 'Location', 'SouthEastOutside');
title('Conductivity vs concentration for Cl/Cl2 by electrolyte');
xlabel('Concentration');
ylabel('Conductivity');

%%
% g. Find the electrolyte that shows the largest change (Max-Min values)
% in conductivity across concentrations and plot its conductivity values
% (in blue). In the same graph plot also the conductivity values for the
% electrolyte which changed the least (in red). Add axis labels a title 
% and a legend.

t_conductivity = conductivity';
min_cond_byElectrolyte    = min(t_conductivity);
max_cond_byElectrolyte    = max(t_conductivity);
diff_cond  = max_cond_byElectrolyte - min_cond_byElectrolyte;

[min_diff_value, min_diff_index] = min(diff_cond);
[max_diff_value, max_diff_index] = max(diff_cond);

figure;
hold on;
plot(concentration, t_conductivity(:, min_diff_index), 'r', 'LineWidth', 2);
plot(concentration, t_conductivity(:, max_diff_index), 'b', 'LineWidth', 2);
hold off;
legend([electrolyte_names(min_diff_index), electrolyte_names(max_diff_index)], 'Location', 'SouthEastOutside');
title('Max and min conductivity difference by electrolyte');
xlabel('Concentration')
ylabel('Conductivity');

%%
% h. Load the file pH_in_solution.mat (after reviewing the spread sheet pH_in_solution.xls)
% You should have 3 variables:
% temp – temperature of the solution in which the acidity (pH) was measured. 
% pH – acidity of each solution across different temperatures.
% solution_names – names of the solutions.


load('pH_in_solution.mat');

% Use the function subplot to show two graphs on the same figure: the first
% graph should be pH as a function of temperature for all the acidic 
% solutions (average pH<7) and the second graph should be pH as a function 
% of temperature for all the alkaline solutions (average pH>7). 
% Make sure to set the neutral pH value as the upper/lower ylimits 
% accordingly (acidic/alkaline). Hint: don’t use the value 7 directly, 
% assign it to a variable (i.e. avoid magic numbers).

pH_by_solution = pH';

pH_neutral = 7;
pH_mean = mean(pH_by_solution);
pH_acidic_index = find(pH_mean < pH_neutral);
pH_basic_index = find(pH_mean > pH_neutral);


figure;
subplot(1,2,1);
plot(temp, pH_by_solution(:, pH_acidic_index), 'LineWidth', 2);
title('pH vs. temp for acidic conditions');
xlabel('Temp')
ylabel('pH');
legend(solution_names(pH_acidic_index), 'Location', 'SouthEastOutside');

subplot(1,2,2);
plot(temp, pH_by_solution(:, pH_basic_index), 'LineWidth', 2);
title('pH vs. temp for basic conditions');
xlabel('Temp')
ylabel('pH');
legend(solution_names(pH_basic_index), 'Location', 'SouthEastOutside');

%% Q2
% In this question we will visualize the electrolyte condcutvity data 
% (from question 1) using the function imagesc.
% Figure 7: Use the imagesc function to plot this data. Don’t forget 
% to add title and axis labels.
 
figure;
imagesc(conductivity);
title('Conductivity heatmap');

% Colorbar + use the function colormap and set the colormap to hot. 
colorbar; colormap hot;

% Remove the “y-axis” ticks.
set(gca, 'YTick', []);
xlabel('Concentration');
ylabel('Electrolytes');


%%
% Figure 8: Create a new figure which is identical to the previous one 
% (figure 7) only with a different color range using the function 
% caxis([0 155]) (see help) .
% What happened to the colors? In what cases will you use this function?  

figure;
imagesc(conductivity);
title('Conductivity heatmap');
% Colorbar + use the function colormap and set the colormap to hot.
% caxis([0 155])
colorbar; colormap hot; caxis([0 155]);
% Remove the “y-axis” ticks.
set(gca, 'YTick', []);
xlabel('Concentration');
ylabel('Electrolytes');

% What happened to the colors? In what cases will you use this function?  
% 
% caxis(V), where V is the two element vector [cmin cmax], sets manual
% scaling of pseudocolor ... in other words values less than cmin or
% greater than cmax will be set to those values, and values falling in
% between will be scaled. This function allows a user to better visualize
% data, and highlight outliers or those passing some threshold.

