classdef MapOptions
    properties (Access = public)
        basisType = BasisTypes.ProbabilistHermite;
        posFuncType = PosFuncTypes.SoftPlus;
        quadType = QuadTypes.AdaptiveSimpson;
        quadAbsTol = 1e-6;
        quadRelTol = 1e-6;
        quadMaxSub = 30;
        quadMinSub = 0;
        quadPts = 5;
        contDeriv = true;
    end

    methods
        function obj = set.basisType(obj,type)
            obj.basisType = type;
        end
        function obj = set.posFuncType(obj,type)
            obj.posFuncType = type;
        end
        function obj = set.quadType(obj,type)
            obj.quadType = type;
        end
        function obj = set.quadAbsTol(obj,value)
            obj.quadAbsTol = value;
        end
        function obj = set.quadRelTol(obj,value)
            obj.quadRelTol = value;
        end
        function obj = set.quadMaxSub(obj,value)
            obj.quadMaxSub = value;
        end
        function obj = set.quadMinSub(obj,value)
            obj.quadMinSub = value;
        end
        function obj = set.quadPts(obj,value)
            obj.quadPts = value;
        end
        function obj = set.contDeriv(obj,value)
            obj.contDeriv = value;
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
        end
    end

end