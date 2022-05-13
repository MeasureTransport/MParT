classdef TestExp < handle
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
  id_ % ID of the session.
end

methods
  function this = TestExp()
  %DATABASE Create a new database.
    this.id_ = TestExp_('new');
  end

  function delete(this)
  %DELETE Destructor.
    TestExp_('delete', this.id_);
  end

  function result = Evaluate1d(this, x)
    result = TestExp_('Evaluate1d',this.id_, x);
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
