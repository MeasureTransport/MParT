classdef CreateTriangular < handle
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
  function this = CreateTriangular(inputDim,outputDim,totalOrder,mapOptions)
  %DATABASE Create a new total order triangular map.
    mexOptions = mapOptions.getMexOptions;
    input_str=['MParT_(',char(39),'newTotalTriMap',char(39),',inputDim,outputDim,totalOrder'];
    for o=1:length(mexOptions)
      input_o=[',mexOptions{',num2str(o),'}'];
      input_str=[input_str,input_o];
    end
    input_str=[input_str,')'];
    this.id_ = eval(input_str);
  
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

  function result = CoeffGrad(this,pts,sens)
    result = MParT_('CoeffGrad',this.id_,pts,sens);
  end

  function result = LogDeterminantCoeffGrad(this,pts)
    result = MParT_('LogDeterminantCoeffGrad',this.id_,pts);
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
