classdef GaussianKLObjective < handle

properties (Access = private)
    id_
end

methods
    function this = GaussianKLObjective(train, test, dim)
        if(nargin==1)
            this.id_ = MParT_('GaussianKLObjective_newTrain',train,0);
        elseif(isinteger(test))
            this.id_ = MParT_('GaussianKLObjective_newTrain',train,test);
        elseif(nargin==2)
            this.id_ = MParT_('GaussianKLObjective_newTrainTest',train,test,0);
        else
            this.id_ = MParT_('GaussianKLObjective_newTrainTest',train,test,dim);
        end
    end

    function delete(this)
        MParT_('MapObjective_delete',this.id_);
    end

    function result = TestError(this, map)
        result = MParT_('GaussianKLObjective_TestError',this.id_,map.get_id());
    end

    function result = TrainError(this, map)
        result = MParT_('GaussianKLObjective_TrainError',this.id_,map.get_id());
    end

    function result = TrainCoeffGrad(this, map)
        result = zeros(map.numCoeffs,1);
        MParT_('GaussianKLObjective_TrainCoeffGrad', this.id_, map.get_id(),result);
    end

    function result = get_id(this)
        result = this.id_;
    end

end
end