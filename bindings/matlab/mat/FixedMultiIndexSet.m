classdef FixedMultiIndexSet < handle

properties (Access = private)
  id_
end


methods
  function this = FixedMultiIndexSet(varargin)
    if(nargin==1)
        this.id_ = MParT_('FixedMultiIndexSet_fromMultiIndexSet', varargin{1}.get_id());
    else
        this.id_ = MParT_('FixedMultiIndexSet_newTotalOrder',varargin{1},varargin{2});
    end
  end

  function delete(this)
  %DELETE Destructor.
    MParT_('FixedMultiIndexSet_delete', this.id_);
  end

  function result = MaxDegrees(this)
    result = MParT_('FixedMultiIndexSet_MaxDegrees', this.id_);
  end

  function Print(this)
    MParT_('FixedMultiIndexSet_Print', this.id_);
  end 

  function size = Size(this)
    size = MParT_('FixedMultiIndexSet_Size', this.id_);
  end 

  function ind = MultiToIndex(this, multi)
    ind = MParT_('FixedMultiIndexSet_MultiToIndex', this.id_, multi);
  end 

  function multi = IndexToMulti(this, multi)
    multi = MParT_('FixedMultiIndexSet_IndexToMulti', this.id_, multi);
  end 

  function dim = Length(this)
    dim = MParT_('FixedMultiIndexSet_Length', this.id_);
  end 

  function result = get_id(this)
    result = this.id_;
  end

end

end
