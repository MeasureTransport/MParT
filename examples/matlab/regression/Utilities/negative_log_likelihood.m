function [L, dcL] = negative_log_likelihood(ref,S, X, coeff_idx)

    % define delta (regularization term) - add small diagonal term
    delta = 1e-9;
    S.setCoeffs(coeff_idx) 
    
    % evaluate objective
    Sk = S.Evaluate(X) + delta*X(:,end);
    logDet = S.GradCoeff(X) + delta;
    % evaluate log_pi(x)
    L = ref.LogPdf(Sk) + logDet;
    L = -1 * mean(L,1);
    
    % evaluate gradient
    if (nargout > 1)
        % evaluate \nabla_c S, \nabla_c_xd S
        dcSk = S.CoeffJacobian(X);
        dcdxSk = S.grad_coeff_grad_xd(X);
        % evaluate \nabla_c log_pi(x)
        dcL = ref.GradLogPdf(Sk) .* dcSk + dcdxSk ./ logDet;
        dcL = -1 * mean(dcL,1);
    end

end
