classdef GaussianKLObjective < handle

properties (Access = private)
    id_
end

methods
    function this = GaussianKLObjective(varargin)
        if(nargin==1)
            this.id_ = MParT_('GaussianKLObjective_newTrain',varargin{1});
        else
            this.id_ = MParT_('GaussianKLObjective_newTrainTest',varargin{1},varargin{2});
        end
    end

    function delete(this)
        MParT_('GaussianKLObjective_delete',this.id_);
    end

    function result = TestError(this, map)
        result = MParT_('GaussianKLObjective_TestError',this.id_,map.get_id());
    end
    
    function result = TrainError(this, map)
        result = MParT_('GaussianKLObjective_TrainError',this.id_,map.get_id());
    end

    function result = TrainCoeffGrad(this, map)
        result = zeros(map.numCoeffs);
        MParT_('GaussianKLObjective_TrainCoeffGrad', this.id_, map.get_id());
    end
    
    function result = get_id(this)
        result = this.id_;
    end

end
end