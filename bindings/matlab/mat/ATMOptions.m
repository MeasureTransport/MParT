classdef ATMOptions < TrainOptions & MapOptions
    properties (Access = public)
        maxPatience = 10;
        maxSize = 10;
        maxDegrees = MultiIndex(0);
    end
    methods
        function obj = set.maxPatience(obj,value)
            obj.maxPatience = value;
        end
        function obj = set.maxSize(obj,value)
            obj.maxSize = value;
        end
        function obj = set.maxDegrees(obj,value)
            obj.maxDegrees = value;
        end
        function optionsArray = getMexOptions(obj)
            getMexOptions@MapOptions(obj);
            optionsArray{16+1} = char(obj.opt_alg);
            optionsArray{16+2} = obj.opt_stopval;
            optionsArray{16+3} = obj.opt_ftol_rel;
            optionsArray{16+4} = obj.opt_ftol_abs;
            optionsArray{16+5} = obj.opt_xtol_rel;
            optionsArray{16+6} = obj.opt_xtol_abs;
            optionsArray{16+7} = obj.opt_maxeval;
            optionsArray{16+8} = obj.opt_maxtime;
            optionsArray{16+9} = obj.verbose;
            optionsArray{16+9+1} = obj.maxPatience;
            optionsArray{16+9+2} = obj.maxSize;
            optionsArray{16+9+3} = obj.maxDegrees.get_id();
        end
    end
end