function [L] = negative_log_likelihood(ref,S, X, coeff_idx)

    % define delta (regularization term) - add small diagonal term
    delta = 1e-9;
    S.setCoeffs(coeff_idx) 
    
    % evaluate objective
    Sk = S.Evaluate(X) + delta*X(:,end);
    logDet = S.GradCoeff(X) + delta;
    % evaluate log_pi(x)
    L = ref.LogPdf(Sk) + logDet;
    L = -1 * mean(L,1);
    
end
