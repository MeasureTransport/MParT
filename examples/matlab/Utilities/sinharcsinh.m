function [x] = sinharcsinh(z,loc,scale,skew,tail)
    
%     To make skewed and/or non-Gaussian tailed test distributions
%     skew \in R, skew > 0 leads to positive (right tilted) skew, skew < 0 leads to negative (left tilted) skew
%     tail > 0, tail < 1 leads to light tails, tail > 1 leads to heavy tails.
%     skew = 0, tail = 1 leads to affine function x = loc + scale*z
%     See for more info: Jones, M. Chris, and Arthur Pewsey. "Sinh-arcsinh distributions." Biometrika 96.4 (2009): 761-780.
%    
    f0 = sinh(tail*asinh(2));
    f = (2/f0)*sinh(tail*(asinh(z) + skew));
    x = loc + scale*f;
end