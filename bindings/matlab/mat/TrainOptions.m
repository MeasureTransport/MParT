classdef TrainOptions
    properties (Access = public)
        opt_alg = "LD_LBFGS"
        opt_stopval = log(0)
        opt_ftol_rel = 1e-3
        opt_ftol_abs = 1e-3
        opt_xtol_rel = 1e-3
        opt_xtol_abs = 1e-3
        opt_maxeval = 30
        opt_maxtime = 100
        verbose = false
    end

    methods
        function obj = set.opt_alg(obj,value)
            obj.opt_alg = value;
        end
        function obj = set.opt_stopval(obj,value)
            obj.opt_stopval = value;
        end
        function obj = set.opt_ftol_rel(obj,value)
            obj.opt_ftol_rel = value;
        end
        function obj = set.opt_ftol_abs(obj,value)
            obj.opt_ftol_abs = value;
        end
        function obj = set.opt_xtol_rel(obj,value)
            obj.opt_xtol_rel = value;
        end
        function obj = set.opt_xtol_abs(obj,value)
            obj.opt_xtol_abs = value;
        end
        function obj = set.opt_maxeval(obj,value)
            obj.opt_maxeval = value;
        end
        function obj = set.opt_maxtime(obj,value)
            obj.opt_maxtime = value;
        end
        function obj = set.verbose(obj,value)
            obj.verbose = value;
        end
        function optionsArray = getMexOptions(obj)
            optionsArray{1} = char(obj.opt_alg);
            optionsArray{2} = obj.opt_stopval;
            optionsArray{3} = obj.opt_ftol_rel;
            optionsArray{4} = obj.opt_ftol_abs;
            optionsArray{5} = obj.opt_xtol_rel;
            optionsArray{6} = obj.opt_xtol_abs;
            optionsArray{7} = obj.opt_maxeval;
            optionsArray{8} = obj.opt_maxtime;
            optionsArray{9} = obj.verbose;
        end
    end
end