%% 
% HW_4 solution
% Joseph Georgeson  336178264
% Itai Antebi       204817498

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% In HW_4 below we use for loops to plot multiplt plots and generate a
% video out of imags. In Q3 we plot 2 different types of plots on our
% 'original' data (taken from matlab library).
% https://www.mathworks.com/help/stats/sample-data-sets.html
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;
clc;

%% Q1
axes_limits = [[-10 10]; [-5 5]; [-3, 3]];
intervals = [0.4 0.3 0.2 0.1]; 
num_of_plots_in_fig = size(intervals,2); 
for ii = 1:size(axes_limits,1)
    figure;
    for cur_plot_num = 1:num_of_plots_in_fig
        subplot(2,2,cur_plot_num);
        cur_axes_min = axes_limits(ii, 1); 
        cur_axes_max = axes_limits(ii, 2); 
        cur_interval = intervals(cur_plot_num); 
        [x, y] = meshgrid(cur_axes_min:cur_interval:cur_axes_max);
        z = -y .* (x.^2 - y.^2) ./ (x.^2 + y.^2 + eps);
        switch(cur_plot_num)
            case {1, 2}
                s = surf(x,y,z); 
            case 3
                s = surf(x,y,z, 'EdgeColor', 'none'); 
            case 4
                contourf(z); 
        end
    end
end

%% Q2
dir_list = dir('images_dir\*.tif'); % for unix dir_list = dir('images_dir/*.tif');
n_frames = size(dir_list,1); 
M = moviein(n_frames);
figure; 
for ii_image = 1:n_frames
    Im = imread(['images_dir\', dir_list(ii_image).name]); % for unix Im = imread(['images_dir/', dir_list(ii_image).name]);
    % The adjustment is used to increase the contrast of the image. 
    % This helps visualize the differences in the image and avoids showing
    % an image which is entirely dark / bright in which the information is
    % not easily seen by the naked eye. 
    Im = imadjust(Im);
    imshow(Im); 
    M(ii_image) = getframe; 
end
movie(M); 

%% Q3
% Our mat file consists of hospital records of 100 people. Records contain
% information such as Sex, Age, Weight, Smoker and Blood Preassure. We
% explore the effect of different parameters on systolic blood pressure.

% Load the data from a file
load('HW4_336178264_204817498.mat'); 
systolic_blood_pressure = hospital.BloodPressure(:, 1); 
% Find indices of different groups
male_indices = find(hospital.Sex == 'Male');
female_indices = find(hospital.Sex == 'Female');
smoking_indices = find(hospital.Smoker); 
non_smoking_indices = find(~hospital.Smoker); 
male_smoking_indices = intersect(male_indices, smoking_indices); 
male_non_smoking_indices = intersect(male_indices, non_smoking_indices); 
female_smoking_indices = intersect(female_indices, smoking_indices); 
female_non_smoking_indices = intersect(female_indices, non_smoking_indices);
% Create two subplots
% First - dot plot of different groups
figure;
subplot(1,2,1);
plot(hospital.Weight(male_non_smoking_indices,1), systolic_blood_pressure(male_non_smoking_indices,1), 'b.')
hold on; 
plot(hospital.Weight(male_smoking_indices,1), systolic_blood_pressure(male_smoking_indices,1), 'bx')
plot(hospital.Weight(female_non_smoking_indices,1), systolic_blood_pressure(female_non_smoking_indices,1), 'r.')
plot(hospital.Weight(female_smoking_indices,1), systolic_blood_pressure(female_smoking_indices,1), 'rx')
xlabel('Weight (lb)', 'FontSize', 14);
ylabel('Systolic Blood Pressure (mmHg)', 'FontSize', 14);
title('Systolic Blood Pressure vs Weight of different groups', 'FontSize', 16);
legend('Male Non Smoking', 'Male Smoking', 'Female Non Smoking', 'Female Smoking', 'Location', 'SouthEastOutside');
% Second - bar plot of elevated blood preasure
elevated_systolic_blood_pressure = 120; 
elevated_systolic_blood_pressure_indices = find(systolic_blood_pressure>elevated_systolic_blood_pressure); 
male_non_smoking_elevated_percent = size(intersect(elevated_systolic_blood_pressure_indices, male_non_smoking_indices),1) / size(male_non_smoking_indices,1); 
male_smoking_elevated_percent = size(intersect(elevated_systolic_blood_pressure_indices, male_smoking_indices),1) / size(male_smoking_indices,1); 
female_non_smoking_elevated_percent = size(intersect(elevated_systolic_blood_pressure_indices, female_non_smoking_indices),1) / size(female_non_smoking_indices,1); 
female_smoking_elevated_percent = size(intersect(elevated_systolic_blood_pressure_indices, female_smoking_indices),1) / size(female_smoking_indices,1); 
elevated_percentage_data = [...
    100 * male_non_smoking_elevated_percent ...
    100 * male_smoking_elevated_percent ...
    100 * female_non_smoking_elevated_percent ...
    100 * female_smoking_elevated_percent ];
subplot(1,2,2);
bar(1, elevated_percentage_data(1), 'b');
hold on; 
bar(2, elevated_percentage_data(2), 'b');
bar(3, elevated_percentage_data(3), 'r');
bar(4, elevated_percentage_data(4), 'r');
set(gca, 'XTick',1:length(elevated_percentage_data));
set(gca, 'XTickLabel', { ...
    'Male Non Smoking' ; ...
    'Male Smoking' ; ...
    'Female Non Smoking' ; ...
    'Female Smoking' ...
    });
set(gca, 'XTickLabelRotation', 90);
title('Percentage of High Blood Preassure of different groups');
xlabel('Group', 'FontSize', 14);
ylabel('Percentage of High Systolic Blood Preassure ', 'FontSize', 14);

% We think that the dot plot visualizes the data in a more suitable way. 
% The bar plot might lead to the same conclusion that smoking increases
% blood preassure, but the dot plot also visualizes the fact the weight and
% sex has little to no effect on blood pressure. 
