% Introduction to Matlab & Data Analysis 2021 semester B
% HW1_336178264_204817498
% Students: Joseph Georgeson, Itai Antebi

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Solution for exercise #1 (question 1-3)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
clc;

%% question 1

% Q1.a,b,c: okay

% Q1.d: displaying the current directory
disp(pwd);

% Q1.e,f:
% We were prompted to either change directories or add this
% directory to PATH. We added the directory to PATH and the script
% ran fine.

% Q1.g:
% After running "rmpath('C:\matlab_course\HW1');", the directory is removed 
% from PATH and we get the same error we had in Q1.e

% Q1.h:
% After running "cd 'C:\matlab_course\HW1';", the script is in our current
% directory, so Matlab finds it even if it is not in PATH and we don't get
% an error

%% question 2

% Q2.a:
Resistance = 5; % [ohm]

% Q2.b: 
Current = 3; % [amp]

% Q2.c: 
Voltage = Resistance * Current; % [volts]

% Q2.d: 
Display_Voltage_String = sprintf('The voltage is: %d', Voltage);
disp(Display_Voltage_String);

% Q2.e: 
% Current type is double

% Q2.f: 
who;

% Q2.g: 
save('save_voltage.mat', 'Voltage');

% Q2.h: 
% We ran "clear Voltage;" in the command line

% Q2.i: 
who; 
% There are 3 variables in the working space
% Current, Display_Voltage_String, Resistance

% Q2.j: 
% We ran "load('save_voltage.mat', 'Voltage');" in the command line.
% There are 4 variables in the working space
% Current, Display_Voltage_String, Resistance, Voltage

% Q2.k: 
% We ran "clear;" in the command line.
% There are 0 variables in the working space

% Q2.l: 
% We ran "Current =7; Resistance =5;" in the command line.
% There are 2 variables in the working space
% Current, Resistance

% Q2.m: 
% 15, because we redefined Current and Resistance. We need to change
% Current and Resistance to 7 & 5 in the script.

% Q2.n: 
% We get the error "Unrecognized function or variable 'Resistance'."
% This is because after declearing the variables the function - "clear"
% earased them. When the script attempted to access them later on - Matlab 
% couldn't resolve their names. 

% Q2.o: 
% clc/clear at the beginning of a script creates a "fresh" environment so
% so nothing might interefere.
% For example, if before the beginning of the script the environment has 
% been corrupted in the following manner "disp = 1;", then running our
% script without 'clear' in the beggining will cause a run time error.
% Adding 'clear' to the begginig of the script will prevent this run time
% error. 

%% question 3

% Q3.a: 
% this code multiplies two numbers and displays the result: 
num1=10;
num2=13; 
num3=4;

disp(['If we multiply	',...
num2str(num1), ' and ', num2str(num2), ' we get: '... 
num2str(num1*num2)]);

% Q3.b: 
% We removed the second (redundant) close parenthesis that was marked red.
% It ran with an error: "Unrecognized function or variable 'num'."

% Q3.c: 
% The warning is "The value assigned to variable 'num2' might be unused". 
%
% Syntax errors prevent Matlab from running a line, as it is not in the
% programming format that Matlab understands. 
% Warning are used to indicate to the programmer that he might be mistaken,
% but that does not prevent from Matlab the ability to run the code. 
%
% We changed the last variable name from 'num2' to 'num3' as there was a
% 'num2' declared before him. The warning message was very indicative. 

% Q3.d: 
% The problem is that the code refers to the variable 'num' that was never
% declared or assigned a value. 
% We changed the variable from 'num' to 'num1' and that fixed the error. 

% Q3.e: 
% Syntax errors prevent Matlab from running a line, as it is not in the
% programming format that Matlab understands. Thus, Matlab recognizes the
% error before running the code and doesn't allow the script to run at all.
%
% Run time error are errors the occur during run time. Matlab cannot know
% in advance that the program will not be able to finish, so the error is
% detected only after some of the code has been executed during Matlab's
% attempt to run a specific line he cannot in his current state. 
