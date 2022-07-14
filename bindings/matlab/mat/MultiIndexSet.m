classdef MultiIndexSet < handle
%DATABASE Example usage of the mexplus development kit.
%
% This class definition gives an interface to the underlying MEX functions
% built in the private directory. It is a good practice to wrap MEX functions
% with Matlab script so that the API is well documented and separated from
% its C++ implementation. Also such a wrapper is a good place to validate
% input arguments.
%
% Build
% -----
%
%    make
%
% See `make.m` for details.
%

properties (Access = private)
  id_
end

methods(Static)
    function mset = CreateTotalOrder(dim, totalOrder)
        mset = MultiIndexSet(dim,totalOrder);
    end
end

methods
  function this = MultiIndexSet(varargin)
    if(nargin==2)
        this.id_ = MParT_('MultiIndexSet_newTotalOrder', varargin{1},varargin{2});
    else
        this.id_ = MParT_('MultiIndexSet_newEigen',varargin{1});
    end
  end

  function delete(this)
  %DELETE Destructor.
    MParT_('MultiIndexSet_delete', this.id_);
  end

  function result = MaxOrders(this)
    result = MParT_('MultiIndexSet_MaxOrders', this.id_);
  end

  function result = get_id(this)
    result = this.id_;
  end

  function fixed_mset = Fix(this)
    fixed_mset = FixedMultiIndexSet(this);
  end
  
end

end
