function optimize_LS(map,X,Y)

obj = @(a) objective_LS(map,X,Y,a);
a0 = map.Coeffs();
options = optimoptions('fminunc','SpecifyObjectiveGradient', false, 'Display', 'none');
[~] = fminunc(obj, a0, options);

end

function [L,dwL] = objective_LS(map,X,Y,coeffs)

map.SetCoeffs(coeffs);
eval = map.Evaluate(X);
DY = (eval-Y);
L = 0.5*mean(DY.^2);
% evaluate gradient
if (nargout > 1)
    sens = DY;
    grad_a_map = map.CoeffGrad(X,sens);
    dwL = mean(grad_a_map,2);
end

end