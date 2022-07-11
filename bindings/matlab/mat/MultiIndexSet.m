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

methods
  function this = MultiIndexSet(multis)
  %DATABASE Create a new database.
    this.id_ = MParT_('newMutliIndexSetEigen',multis);
  end

  function delete(this)
  %DELETE Destructor.
    MParT_('deleteMultiIndexSet', this.id_);
  end

  function result = MaxOrders(this)
    result = MParT_('MaxOrders', this.id_);
  end

  function result = get_id(this)
    result = this.id_;
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