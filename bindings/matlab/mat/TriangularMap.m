classdef TriangularMap < handle
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
  function this = TriangularMap(list_id)
  %DATABASE Create a new database.
    this.id_ = MParT_('newTriMap',list_id);
  end

 function delete(this)
  %DELETE Destructor.
    MParT_('deleteMap', this.id_);
  end

  function SetCoeffs(this,coeffs)
    MParT_('SetCoeffs',this.id_,coeffs(:));
  end

  function result = Coeffs(this)
    result = MParT_('Coeffs',this.id_);
  end

  function result = numCoeffs(this)
    result = MParT_('numCoeffs',this.id_);
  end

  %function result = Evaluate(this,pts)
  %  result = zeros(1,size(pts,2));
  %  MParT_('Evaluate',this.id_,pts,result);
  %end

  function result = Evaluate(this,pts)
    result = MParT_('Evaluate',this.id_,pts);
  end

  function result = LogDeterminant(this,pts)
    result = MParT_('LogDeterminant',this.id_,pts);
  end

  function result = Inverse(this,x1,r)
    result = MParT_('Inverse',this.id_,x1,r);
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
