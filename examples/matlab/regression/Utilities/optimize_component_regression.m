function optimize_component_regression(map,X,Y)

obj = @(a) objective_LS(map,X,Y,a);
a0 = map.Coeffs();
options = optimoptions('fminunc','SpecifyObjectiveGradient', false, 'Display', 'none');
[~] = fminunc(obj, a0, options);

end

function L = objective_LS(map,X,Y,coeffs)

map.setCoeffs(coeffs);
eval = map.Evaluate(X);
L = mean((eval-Y).^2);
end