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
        basisLB = log(0);
        basisUB = 1.0/0.0;
        basisNorm = true;
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
            optionsArray{10} = obj.basisLB;
            optionsArray{11} = obj.basisUB;
            optionsArray{12} = obj.basisNorm;
        end

        function res = eq(obj1, obj2)
            res = true;
            res = res && isequal(obj1.basisType,   obj2.basisType);
            res = res && isequal(obj1.basisLB,     obj2.basisLB);
            res = res && isequal(obj1.basisUB,     obj2.basisUB);
            res = res && isequal(obj1.basisNorm,   obj2.basisNorm);
            res = res && isequal(obj1.posFuncType, obj2.posFuncType);
            res = res && isequal(obj1.quadType,    obj2.quadType);
            res = res && isequal(obj1.quadAbsTol,  obj2.quadAbsTol);
            res = res && isequal(obj1.quadRelTol,  obj2.quadRelTol);
            res = res && isequal(obj1.quadMaxSub,  obj2.quadMaxSub);
            res = res && isequal(obj1.quadMinSub,  obj2.quadMinSub);
            res = res && isequal(obj1.quadPts,     obj2.quadPts);
            res = res && isequal(obj1.contDeriv,   obj2.contDeriv);
        end

        function Serialize(obj,filename)
            MParT_('MapOptions_Serialize',filename, char(obj.basisType),  ...
             char(obj.posFuncType), char(obj.quadType), obj.quadAbsTol,   ...
             obj.quadRelTol, obj.quadMaxSub, obj.quadMinSub, obj.quadPts, ...
             obj.contDeriv, obj.basisLB, obj.basisUB, obj.basisNorm)
        end

        function obj = Deserialize(obj,filename)
            options_len = 12;
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
            obj.basisType = optionsArray{1};
            obj.posFuncType = optionsArray{2};
            obj.quadType = optionsArray{3};
            obj.quadAbsTol = optionsArray{4};
            obj.quadRelTol = optionsArray{5};
            obj.quadMaxSub = optionsArray{6};
            obj.quadMinSub = optionsArray{7};
            obj.quadPts = optionsArray{8};
            obj.contDeriv = optionsArray{9};
            obj.basisLB = optionsArray{10};
            obj.basisUB = optionsArray{11};
            obj.basisNorm = optionsArray{12};
        end
    end

end