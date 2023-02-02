classdef TrainOptions < TrainOptions & MapOptions
    properties (Access = public)
        maxPatience = 10;
        maxSize = 10;
        maxDegrees;
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
            optionsArray{1} = char(obj.basisType);
            optionsArray{2} = char(obj.posFuncType);
            optionsArray{3} = char(obj.quadType);
            optionsArray{4} = obj.quadAbsTol;
            optionsArray{5} = obj.quadRelTol;
            optionsArray{6} = obj.quadMaxSub;
            optionsArray{7} = obj.quadMinSub;
            optionsArray{8} = obj.quadPts;
            optionsArray{9} = obj.contDeriv;
            optionsArray{10} = obj.basisLB;
            optionsArray{11} = obj.basisUB;
            optionsArray{12} = obj.basisNorm;
            optionsArray{12+1} = char(obj.opt_alg);
            optionsArray{12+2} = obj.opt_stopval;
            optionsArray{12+3} = obj.opt_ftol_rel;
            optionsArray{12+4} = obj.opt_ftol_abs;
            optionsArray{12+5} = obj.opt_xtol_rel;
            optionsArray{12+6} = obj.opt_xtol_abs;
            optionsArray{12+7} = obj.opt_maxeval;
            optionsArray{12+8} = obj.opt_maxtime;
            optionsArray{12+9} = obj.verbose;
        end
    end
end
