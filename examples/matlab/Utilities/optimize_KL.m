function optimize_KL(map,ref,X)

obj = @(a) negative_log_likelihood(map,ref,X,a);
a0 = map.Coeffs();

options = optimoptions('fminunc','SpecifyObjectiveGradient', false, 'Display', 'none');
[~] = fminunc(obj, a0, options);

end

function [L,dwL] = negative_log_likelihood(map,ref,X,coeffs)

map.SetCoeffs(coeffs);
eval = map.Evaluate(X);
log_det = map.LogDeterminant(X);

L = -mean(ref.LogPdf(eval') + log_det);

% evaluate gradient
if (nargout > 1)
    % evaluate \nabla_c S, \nabla_c_xd S
    sens = ref.GradLogPdf(eval');
    grad_a_ref = map.CoeffGrad(X,sens);
    grad_a_log_det = map.LogDeterminantCoeffGrad(X);
    dwL = -1 * mean(grad_a_ref+grad_a_log_det,2);
end

end