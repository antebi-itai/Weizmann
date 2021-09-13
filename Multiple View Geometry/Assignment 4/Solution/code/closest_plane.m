function [plane] = closest_plane(X)
 % (as in 2.b) substitute to the least squares problem without d
 Xbar = mean(X, 2); 
 Xtilde = X - Xbar;
 xbar = Xbar(1);
 ybar = Xbar(2);
 zbar = Xbar(3);
 % (as in 2.b) construct the matrix whos eigenvector is a solution to the
 % least square problem
 M = Xtilde(1:3,:)*Xtilde(1:3 ,:)';
 % (as in 2.b) the solution to the least squares problem is the eigenvector
 % corresponding to the smallest eigenvalue
 [V, D] = eig(M);
 sol = V(:, 1);
 % extract a,b,c from the solution
 a = sol(1);
 b = sol(2);
 c = sol(3);
 % (as in 2.a) compute d from the original least squares problem
 d = -(a*xbar + b*ybar + c*zbar);
 
 plane = [a;b;c;d];
 plane = plane ./ norm(plane(1:3));

end