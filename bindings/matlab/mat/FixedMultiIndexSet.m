classdef FixedMultiIndexSet < handle
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
  function this = FixedMultiIndexSet(dim,maxOrder)
  %DATABASE Create a new database.
    this.id_ = BigLibrary_('newFixedMultiIndexSet', dim, maxOrder);
  end

  function delete(this)
  %DELETE Destructor.
    BigLibrary_('deleteFixedMultiIndexSet', this.id_);
  end

  function result = MaxDegrees(this)
    result = BigLibrary_('MaxDegrees',this.id_);
  end

  function result = get_id(this)
    result = this.id_;
  end

  function [result,mset_id] = MaxDegrees2(this,mset2)
    [result,mset_id] = BigLibrary_('MaxDegrees2',this.id_,mset2.id_);
  end

  function result = IndexToMulti(this, ind)
    result = BigLibrary_('IndexToMulti', this.id_, ind);
  end

  function result = MultiToIndex(this, ind)
    result = BigLibrary_('MultiToIndex', this.id_, ind);
  end

  function Print(this)
     BigLibrary_('Print', this.id_);
  end

  function result = Size(this)
    result = BigLibrary_('Size', this.id_);
  end

  function result = dim(this)
    result = BigLibrary_('dim', this.id_);
  end

  function result = isCompressed(this)
    result = BigLibrary_('isCompressed', this.id_);
  end
  
end

methods (Static)
  function environment = getEnvironment()
  %GETENVIRONMENT Get environment info.
    environment = mexmpart('getEnvironment');
  end

  function setEnvironment(environment)
  %SETENVIRONMENT Set environment info.
    mexmpart('setEnvironment', environment);
  end
end

end
