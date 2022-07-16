classdef Banana

    properties
        X1 = Normal(0,0.5)
        X2_noise = Normal(0,0.1)
    end

    methods
        % -----------------------------------------------------------------
        function check_inputs(~, x)
            assert(size(x,2) == 2)
        end
        % -----------------------------------------------------------------
        function logpi = LogPdf(self, x)
            logpi_1 = self.X1.log_pdf(x(:,1));
            logpi_2 = self.X2_noise.log_pdf(x(:,2) - x(:,1).^2);
            logpi = logpi_1 + logpi_2;
        end %endFunction
        % -----------------------------------------------------------------
        function grad_logpi = GradLogPdf(self, x)
            grad1 = -x(1,:)/(self.X1.sigma)^2 + (2*x(1,:).*(x(2,:)-x(1,:).^2))/(self.X2_noise.sigma)^2 ;
            grad2 = (x(1,:).^2-x(2,:))/(self.X1.sigma)^2;
            grad_logpi = [grad1;grad2];
        end %endFunction
        % -----------------------------------------------------------------
        function x = sample(self, N)
            x1 = self.X1.sample(N);
            x2 = x1.^2 + self.X2_noise.sample(N);
            x = [x1,x2];
            x=x';
        end %endFunction
        % -----------------------------------------------------------------
    end

end
