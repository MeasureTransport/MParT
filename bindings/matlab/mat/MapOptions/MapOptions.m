classdef MapOptions
    properties (Access = public)
        basisType = BasisTypes.ProbabilistHermite;
        sigmoidType = SigmoidTypes.Logistic;
        edgeType = EdgeTypes.SoftPlus;
        posFuncType = PosFuncTypes.SoftPlus;
        quadType = QuadTypes.AdaptiveSimpson;
        quadAbsTol = 1e-6;
        quadRelTol = 1e-6;
        quadMaxSub = 30;
        quadMinSub = 0;
        edgeShape = 1.5;
        quadPts = 5;
        contDeriv = true;
        basisLB = log(0);
        basisUB = 1.0/0.0;
        basisNorm = true;
        nugget = 0.0;
    end

    methods
        function obj = set.basisType(obj,type)
            obj.basisType = type;
        end
        function obj = set.basisLB(obj,value)
            obj.basisLB = value;
        end
        function obj = set.basisUB(obj,value)
            obj.basisUB = value;
        end
        function obj = set.basisNorm(obj,value)
            obj.basisNorm = value;
        end
        function obj = set.posFuncType(obj,type)
            obj.posFuncType = type;
        end
        function obj = set.sigmoidType(obj,type)
            obj.sigmoidType = type;
        end
        function obj = set.edgeShape(obj,value)
            obj.edgeShape = value;
        end
        function obj = set.edgeType(obj,value)
            obj.edgeType = value;
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
        function obj = set.nugget(obj,value)
            obj.nugget = value;
        end
        function optionsArray = getMexOptions(obj)
            optionsArray{1} = char(obj.basisType);
            optionsArray{2} = char(obj.sigmoidType);
            optionsArray{3} = char(obj.edgeType);
            optionsArray{4} = char(obj.posFuncType);
            optionsArray{5} = char(obj.quadType);
            optionsArray{6} = obj.quadAbsTol;
            optionsArray{7} = obj.quadRelTol;
            optionsArray{8} = obj.quadMaxSub;
            optionsArray{9} = obj.quadMinSub;
            optionsArray{10} = obj.edgeShape;
            optionsArray{11} = obj.quadPts;
            optionsArray{12} = obj.contDeriv;
            optionsArray{13} = obj.basisLB;
            optionsArray{14} = obj.basisUB;
            optionsArray{15} = obj.basisNorm;
            optionsArray{16} = obj.nugget;
        end

        function res = eq(obj1, obj2)
            res = true;
            res = res && isequal(obj1.basisType,   obj2.basisType);
            res = res && isequal(obj1.basisLB,     obj2.basisLB);
            res = res && isequal(obj1.basisUB,     obj2.basisUB);
            res = res && isequal(obj1.basisNorm,   obj2.basisNorm);
            res = res && isequal(obj1.posFuncType, obj2.posFuncType);
            res = res && isequal(obj1.sigmoidType, obj2.sigmoidType);
            res = res && isequal(obj1.edgeType,    obj2.edgeType);
            res = res && isequal(obj1.edgeShape,   obj2.edgeShape);
            res = res && isequal(obj1.quadType,    obj2.quadType);
            res = res && isequal(obj1.quadAbsTol,  obj2.quadAbsTol);
            res = res && isequal(obj1.quadRelTol,  obj2.quadRelTol);
            res = res && isequal(obj1.quadMaxSub,  obj2.quadMaxSub);
            res = res && isequal(obj1.quadMinSub,  obj2.quadMinSub);
            res = res && isequal(obj1.quadPts,     obj2.quadPts);
            res = res && isequal(obj1.contDeriv,   obj2.contDeriv);
            res = res && isequal(obj1.nugget,      obj2.nugget);
        end

        function Serialize(obj,filename)
            MParT_('MapOptions_Serialize',filename, char(obj.basisType),  ...
            char(obj.sigmoidType), char(obj.edgeType), char(obj.posFuncType), char(obj.quadType), ...
             obj.quadAbsTol, obj.quadRelTol, obj.quadMaxSub, obj.quadMinSub, obj.edgeShape, ...
             obj.quadPts, obj.contDeriv, obj.basisLB, obj.basisUB, obj.basisNorm, obj.nugget)
        end

        function obj = Deserialize(obj,filename)
            options_len = 16;
            optionsArray = cell(options_len,1);
            input_str = [']=MParT_(',char(39),'MapOptions_Deserialize',char(39),',filename);'];
            for i=options_len:-1:1
                comma_str = ',';
                if i == 1
                    comma_str='[';
                end
                add_str = [comma_str,'optionsArray{',num2str(i),'}'];
                input_str = [add_str, input_str];
            end
            eval(input_str);

            obj.basisType    = optionsArray{1};
            obj.sigmoidType  = optionsArray{2};
            obj.edgeType     = optionsArray{3};
            obj.posFuncType  = optionsArray{4};
            obj.quadType     = optionsArray{5};
            obj.quadAbsTol   = optionsArray{6};
            obj.quadRelTol   = optionsArray{7};
            obj.quadMaxSub   = optionsArray{8};
            obj.quadMinSub   = optionsArray{9};
            obj.edgeShape    = optionsArray{10};
            obj.quadPts      = optionsArray{11};
            obj.contDeriv    = optionsArray{12};
            obj.basisLB      = optionsArray{13};
            obj.basisUB      = optionsArray{14};
            obj.basisNorm    = optionsArray{15};
            obj.nugget       = optionsArray{16};
        end
    end

end