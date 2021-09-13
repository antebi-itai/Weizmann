function homogeneousX = pflat(X)
%PFLAT divides the homogeneous coordinates with their last entry 
% for points of any dimensionality
homogeneousX = X ./ X(end,:);
end

