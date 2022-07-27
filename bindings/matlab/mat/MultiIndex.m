classdef MultiIndex < handle
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

methods
  function this = MultiIndex(varargin)
    if(nargin==2) 
      if(isstring(varargin{2})) 
        if(varargin{2}=="id")
          this.id_ = varargin{1};
        end
      else
        this.id_ = MParT_('MultiIndex_newDefault', varargin{1},varargin{2});
      end
    else
        this.id_ = MParT_('MultiIndex_newEigen',varargin{1});
    end
  end

  function delete(this)
  %DELETE Destructor.
    MParT_('MultiIndex_delete', this.id_);
  end

  function result = Vector(this)
    result = MParT_('MultiIndex_Vector', this.id_);
  end

  function result = Sum(this)
    result = MParT_('MultiIndex_Sum', this.id_);
  end

  function result = Max(this)
    result = MParT_('MultiIndex_Max', this.id_);
  end

  function result = Set(this,ind,val)
    result = MParT_('MultiIndex_Set', this.id_,ind-1,val);
  end

  function result = Get(this,ind)
    result = MParT_('MultiIndex_Get', this.id_,ind-1);
  end

  function result = NumNz(this)
    result = MParT_('MultiIndex_NumNz', this.id_);
  end

  function result = String(this)
    result = MParT_('MultiIndex_String', this.id_);
  end

  function result = Length(this)
    result = MParT_('MultiIndex_Length', this.id_);
  end

 % == operator
  function result = eq(this,multi)
    result = MParT_('MultiIndex_Eq',this.id_,multi.get_id());
  end

  % ~= operator
  function result = ne(this,multi)
    result = MParT_('MultiIndex_Ne',this.id_,multi.get_id());
  end

  % < operator
  function result = lt(this,multi)
    result = MParT_('MultiIndex_Lt',this.id_,multi.get_id());
  end

  % > operator
  function result = gt(this,multi)
    result = MParT_('MultiIndex_Gt',this.id_,multi.get_id());
  end

  % >= operator
  function result = Ge(this,multi)
    result = MParT_('MultiIndex_Ge',this.id_,multi.get_id());
  end

  % <= operator
  function result = Le(this,multi)
    result = MParT_('MultiIndex_Le',this.id_,multi.get_id());
  end

  function result = get_id(this)
    result = this.id_;
  end

  
end

end
