function X_3D_out = triangulate(P1n, P2n, x1n, x2n)
    % Use triangulation (with DLT) to compute the 3D-points.
    n = size(x1n, 2);
    X_3D = zeros(4, n);
    lambdas = zeros(2, n);
    for i = 1:n
        x1_i = x1n(:, i); 
        x2_i = x2n(:, i); 
        M_i = zeros(6, 6);
        M_i(1:3, 1:4) = P1n;
        M_i(4:6, 1:4) = P2n;
        M_i(1:3,5) = -x1_i;
        M_i(4:6,6) = -x2_i;
        [Ui, Si, Vi] = svd(M_i);
        Xi = pflat(Vi(1:4, 6));
        lambda_i = Vi(5:6, 6);
        X_3D(:, i) = Xi;
        lambdas(:, i) = lambda_i; 
    end
    X_3D_out = X_3D;
end

