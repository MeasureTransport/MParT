classdef Normal

	% Defines a univariate Gaussian distribution
	% 
	% Methods: log_pdf, grad_log_pdf, hess_log_pdf, sample
	% Date:    July 2019

	properties
		mean   % mean of density
		sigma  % standard deviation of density
	end

	methods 
		function Norm = Normal(mean, sigma)

            if (nargin < 1)
                mean = 0;
            end
            Norm.mean = mean;

            if (nargin < 2)
                sigma = 1;
            end
            Norm.sigma = sigma;

		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
		function logpi = log_pdf(Norm, X)
			logpi = -0.5*log(2*pi*Norm.sigma^2) - 0.5/Norm.sigma^2*(X - Norm.mean).^2;
		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
		function grad_logpi = grad_x_log_pdf(Norm, X)
			grad_logpi = -1/Norm.sigma^2*(X - Norm.mean);
		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
		function hess_logpi = hess_x_log_pdf(Norm, X)
			hess_logpi = -1/Norm.sigma^2*ones(size(X));
		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
		function Z = sample(Norm, N)
			Z = Norm.mean + Norm.sigma*randn(N, 1);
		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
	end %endMethods
    
end %endClass
