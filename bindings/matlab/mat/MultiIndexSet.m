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

  function multi = IndexToMulti(this,activeIndex)
    multi_id = MParT_('MultiIndexSet_IndexToMulti',this.id_,activeIndex-1);
    multi = MultiIndex(multi_id,'id');
  end

  function result = MultiToIndex(this,multi)
    result= MParT_('MultiIndexSet_MultiToIndex',this.id_,multi.get_id());
    result = result + 1;
  end

  function result = Length(this)
    result = MParT_('MultiIndexSet_Length', this.id_);
  end

  function result = MaxOrders(this)
    result = MParT_('MultiIndexSet_MaxOrders', this.id_);
  end

  function multi = at(this,activeIndex)
    multi_id = MParT_('MultiIndexSet_at', this.id_,activeIndex-1);
    multi = MultiIndex(multi_id,'id');
  end

  function varargout = subsref(this,s) %seems dangerous
        switch s(1).type
        case '.'
            % Keep built-in functionality for '.'
            [varargout{1:nargout}] = builtin('subsref', this, s);
        case '()'
            % Keep built-in functionality for '()'
            [varargout{1:nargout}] = builtin('subsref', this, s);
        case '{}'
            activeIndex = s(1).subs{1};
            multi_id = MParT_('MultiIndexSet_subsref',this.id_,activeIndex-1);
            [varargout{1:nargout}] = MultiIndex(multi_id,'id');
        otherwise
            error('Indexing expression invalid.')
        end
  end

  function result = Size(this)
    result = MParT_('MultiIndexSet_Size', this.id_);
  end  

  function result = plus(this,toAdd)
    if strcmp(class(toAdd),'MultiIndexSet')
      MParT_('MultiIndexSet_addMultiIndexSet',this.id_,toAdd.get_id());
    elseif strcmp(class(toAdd),'MultiIndex')
      MParT_('MultiIndexSet_addMultiIndex',this.id_,toAdd.get_id());
    else
      error('Unrecognized type to add to MultiIndexSet')
    end
  end  

  function result = Union(this,mset)
    result = MParT_('MultiIndexSet_Union',this.id_,mset.get_id()); 
  end

  function Activate(this,multi)
    MParT_('MultiIndexSet_Activate',this.id_,multi.get_id()); 
  end

  function result = AddActive(this,multi)
    result = MParT_('MultiIndexSet_AddActive',this.id_,multi.get_id()); 
  end

  function result = Expand(this,activeInd)
    %-1 to keep consitent with matlab ordering
    result = MParT_('MultiIndexSet_Expand',this.id_,activeInd-1); 
  end  

  function result = ForciblyExpand(this,activeInd)
    %-1 to keep consitent with matlab ordering
    result = MParT_('MultiIndexSet_ForciblyExpand',this.id_,activeInd-1);
  end  

  function result = Frontier(this)
    result = MParT_('MultiIndexSet_Frontier',this.id_);
  end  

  function result = StrictFrontier(this)
    result = MParT_('MultiIndexSet_StrictFrontier',this.id_);
  end  

  function result = BackwardNeighbors(this,activeIndex)
    %-1 to keep consitent with matlab ordering
    result = MParT_('MultiIndexSet_BackwardNeighbors',this.id_,activeIndex-1);
  end  

  function result = IsExpandable(this,activeIndex)
    %-1 to keep consitent with matlab ordering
    result = MParT_('MultiIndexSet_IsExpandable',this.id_,activeIndex-1);
  end  

  function result = NumActiveForward(this,activeIndex)
    %-1 to keep consitent with matlab ordering
    result = MParT_('MultiIndexSet_NumActiveForward',this.id_,activeIndex-1);
  end  

  function result = NumForward(this,activeIndex)
    %-1 to keep consitent with matlab ordering
    result = MParT_('MultiIndexSet_NumForward',this.id_,activeIndex-1);
  end  

  function Visualize(this)
    MParT_('MultiIndexSet_Visualize',this.id_);
  end  

  function result = get_id(this)
    result = this.id_;
  end

  function fixed_mset = Fix(this)
    fixed_mset = FixedMultiIndexSet(this);
  end


  
end

end
