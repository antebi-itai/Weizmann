% HW_6 solution
% Joseph Georgeson  336178264
% Itai Antebi       204817498

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% HW6 requires us to extract and replot data from a jpeg from a
% published paper.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
clc;
clear;
close all;

%%

% make sure HW checker skips the ginput part
hw_checker_is_running = true; 
% consts
num_graphs = 3; 
num_points_per_graph = 4; 
x_cal_measure = 80; 
y_cal_measure = 7; 
margin = 0.1;

%%
% 1) Load the .jpg file - image Kjelstrup_et_al_Science_2008.jpg into 
% matlab, using imread(), and plot it using imshow().
input_image = imread('Kjelstrup_et_al_Science_2008.jpg'); 
imshow(input_image);

%%
% 2) Use ginput() to measure the locations of the twelve points in the 
% graph (4 points per category). 
if ~hw_checker_is_running
    [X, Y] = ginput(num_graphs * num_points_per_graph);
    X = reshape(X, num_points_per_graph, num_graphs); 
    Y = reshape(Y, num_points_per_graph, num_graphs);     
end

%%
% 3) Measure a set of calibration points on the x-axis and y-axis so that 
% you'll be able to translate the pixel values you got in part 2) to the 
% same axis values you see in the image.
if ~hw_checker_is_running
    [X_cal, Y_cal] = ginput(2);
    x_zero = X_cal(1);
    y_zero = Y_cal(1); 
    x_cal_measure_location = X_cal(2); 
    y_cal_measure_location = Y_cal(2); 
end

%% 
% 4) Save the two sets of points from (2),(3) in a file called 
% ‘ex6_your_id.mat’, And load it in to your script. 
if ~hw_checker_is_running
    save('ex6_336178264_204817498.mat', ...
        'X', 'Y', ...
        'x_zero', 'y_zero', ...
        'x_cal_measure_location', 'y_cal_measure_location'); 
else
    load('ex6_336178264_204817498.mat'); 
end


%% 
% 1) Use if statement with a variable that indicate whether to use ginput 
% or to load the saved data. 
% DONE !

%% 
% 2) Open a new figure and by using the axes() command create two subplots 
% – the first one should be small and display the original image 
% (use imshow())
width = (1 - 3*margin)/3;
height = 1 - 2*margin; 

fig = figure('units','normalized','outerposition',[0 0 1 1]);
axes('position', [margin, margin, width, height]);
imshow(input_image); 

%%
% 3) The second axes you create should be horizontally double and 
% vertically the same size - Plot on it the data you extracted from the 
% original figure. Make sure to plot the three categories with different 
% symbols as in the original figure. 

x_unit = ((x_cal_measure_location - x_zero) / x_cal_measure); 
y_unit = ((y_zero - y_cal_measure_location) / y_cal_measure); 
calibrated_X = (X - x_zero) / x_unit; 
calibrated_Y = (y_zero - Y) / y_unit; 

axes('position', [2*margin + width, margin, 2*width, height]);
plot(calibrated_X(:, 1), calibrated_Y(:, 1), '-s', 'color', 'k', 'MarkerFaceColor' ,'k');
xlim([0,100]);
ylim([0,8]);
hold on;
plot(calibrated_X(:, 2), calibrated_Y(:, 2), '-s', 'color', 'k', 'MarkerFaceColor' ,'w');
plot(calibrated_X(:, 3), calibrated_Y(:, 3), '-^', 'color', 'k', 'MarkerFaceColor' ,'w');

%%
% 4) Find the linear regression (you can use the ‘fitlm’ function), using 
% all the 12 datapoints, and plot it with a dashed line style. 
lm = fitlm(reshape(calibrated_X, 1, num_graphs * num_points_per_graph), ...
           reshape(calibrated_Y, 1, num_graphs * num_points_per_graph)); 
plot(lm);
% Next, using the regression line you found, predict the data at 40% on 
% the x axis and plot this with a distinct new symbol. 
x_pred = 40; 
y_pred = predict(lm, x_pred); 
plot(x_pred, y_pred, '-x', 'color', 'r', 'MarkerFaceColor' ,'r', 'MarkerSize', 20);
% Add the text “prediction” to the left of this point.
text(x_pred - 8, y_pred, 'prediction'); 

%%
% 5) Annotate the graph: Put a title, an x-label, a y-label and a legend 
% (according to the three categories and the linear regression).
title('Distance vs Position on DV axis'); 
xlabel('Position on DV axis (%)');
ylabel('Distance (m)');
legend('Largest Field Width', 'Population-vector half-width', 'Spatial Autocorrelation', 'location', 'NorthWest')

%%
% 6) Set the xticks and yticks as they are in the original figure.
xticks([0:20:100]);
yticks([0:8]);

%%
% 7) In the code - save your figure in jpeg format – call it ‘ex6_figure_your_id_number.jpg’
saveas(fig, 'ex6_336178264_204817498.jpg')
