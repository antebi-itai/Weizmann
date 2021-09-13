function [  ] = hw5_336178264_204817498()
    % HW_5 solution
    % Joseph Georgeson  336178264
    % Itai Antebi       204817498
    clc;
    clear;
    
    %% question 1.1-8
    % 'doc subplot' tells us what subplot does and its variables
    % subplot(m,n,p) divides the current figure into an m-by-n grid, where
    % m=#row and n=#col, and creates 'axes' in the position specified by p.
    
    function mySubplot(m,n,p)
        
        % error handling:
        if not(isnumeric(m))
            error('m must be numeric');
        end
        if not(isnumeric(n))
            error('n must be numeric');
        end
        if not(isnumeric(p))
            error('p must be numeric');
        end
        if p > m*n
            error('The specified position is beyond the boundaries m x n');
        end
        if m*n > 100
            error('Reconsider your plot design, the dimensions are too much for this assignment');
        end
        
        % set margin sizes between subplots according to number of rows and columns
        if m*n > 90
            margin_size_width=0.005*m;
            margin_size_height=0.005*n;
        elseif m*n > 9
            margin_size_width=0.001*m;
            margin_size_height=0.001*n;
        elseif m*n >= 4
            margin_size_width=0.005*m;
            margin_size_height=0.005*n;
        else % m*n < 4
            margin_size_width=0.04*m;
            margin_size_height=0.04*n;
        end      
        if n == 1
            margin_size_height=0.04*n;
        end
        if m == 1
            margin_size_width=0.04*m;
        end
        
        % calculate size of subplot
        Axis_width  = (1-(n+1)*margin_size_width) /n;
        Axis_height = (1-(m+1)*margin_size_height)/m;

        % calculate position of lower-left corner of subplot
        Pos_width = margin_size_width:(margin_size_width+Axis_width):1-margin_size_width;
        Pos_height = flip(margin_size_height:(margin_size_height+Axis_height):1-margin_size_height);
        [num_m, num_n] = ind2sub([m,n],p);
        x_coords = Pos_width(num_n);
        y_coords = Pos_height(num_m);
        
        % set subplot position
        axes('position',[x_coords, y_coords, Axis_width, Axis_height]);
        
    end
    
    %% question 2.1-2
    % Create a nested function named myFactorial() which calculates the
    % factorial of the number n and prints its value. The factorial is
    % defined as the product n!=∏_(k=1)^n▒k
    % You should program this function using recursion only - without any
    % loops, matrices or other Matlab functions. The nested function should
    % be inside the primary function.
    
    function [out] = myFactorial(n)
        if (n < 0)
            error('n must be non-negative');
        elseif (n == 0)
            out = 1;
        else
            out = n * myFactorial(n-1);
        end
    end
    
    % Call the function with n=8, and print the result.
    clear;
    n=8;
    disp(sprintf('myFactorial(%d)=%d', n, myFactorial(n)));
    
    %% question 3.1-2
    % Create a nested function named calcTailorExp() which gets as input a 
    % number x and the upper limit N (where n=0,…,N), and calculates the 
    % following series: 1+x^1/1!+x^2/2!+x^3/3!+⋯=∑_(n=0)^N▒x^n/n!
    % You should program this function using recursion without any loops, 
    % matrices or other Matlab functions. You should use the factorial 
    % function above.
    
    function[out] =  calcTailorExp(x,n)
        if (n < 0)
            error('n must be non-negative');
        elseif (n == 0)
            out = 1;
        else
            out = ((x^n)/myFactorial(n)) + calcTailorExp(x,n-1);
        end
    end
    
    % Assume x=1,what is the minimal value of N that gives you a good
    % approximation (with an error of less than 0.0005) for an exponent e^1?
    % (This is a matlab course, not a math course.. we want to see the code,
    % not the final result)
    clear;
    n = 0;
    x = 1;
    evaluate_value = exp(1);
    max_error = 0.0005;
    curr_error = max_error + eps;
    while max_error < curr_error
        n = n+1;
        curr_tailor = calcTailorExp(x,n);
        curr_error = abs(evaluate_value - curr_tailor);
    end
    disp(sprintf('To have an error less than %g when calculating e using taylor approximation, n must be >= %d', max_error, n));
    
    %% question 4
    
    % 1. You should have N subplots, where N is the number you got in question 3.2
    clear;
    n=6; % from Q3 above, so we'll have 6 subplots
    x=1; % from Q3 above
    mRows=2; % will define this myself
    nCols=ceil(n / mRows); % derive number of columns from number of subplots and number of rows
    
    figure;
    for p=1:n
        
        % 3. Each approximation will be a different point in the graph.
        % 4. The x-value of each point should be the number n (1 for the
        % first point, 2 for the second and so on..), and the y-value of
        % each point should be the number you will calculate (e^1 or calcTailorExp(n))
        y(p)=calcTailorExp(x,p);
        mySubplot(mRows,nCols,p);
        
        % 2. In each subplot you should plot the real number e^1  as a horizontal
        % line and the approximated numbers based on calcTailorExp(n),
        % where n are all numbers between 1 until the current number of the subplot.
        yline(exp(1), 'b');
        hold on;
        
        % 5. For example, ...
        plot(1:p, y, '-k*', 'MarkerSize', 12);
        
        % 6. Add title to each subplot ‘approximating e with n=4’ in the
        % fourth subplot for example.
        text(2.5, 1, sprintf('Aprrox e with n=%d', p));
        
        % 7. The Y axis of all subplot should be the same and start with
        % zero until ceil(e^1)
        ylim([0, ceil(exp(1))]);
        xlim([0 (n+1)]);
    end

%% close top function
end