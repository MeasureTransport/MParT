classdef ParameterizedFunction < handle
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

  function this = ParameterizedFunction(varargin)
    if(nargin==2)
        if(isstring(varargin{2}))
          if(varargin{2}=="id")
            this.id_=varargin{1};
          end
        end
    elseif(nargin==3)
          outputDim = varargin{1};
          mset = varargin{2};
          mapOptions = varargin{3};
          mexOptions = mapOptions.getMexOptions;
          input_str=['MParT_(',char(39),'ParameterizedFunction_newMap',char(39),',outputDim,mset.get_id()'];
          for o=1:length(mexOptions)
              input_o=[',mexOptions{',num2str(o),'}'];
              input_str=[input_str,input_o];
          end
          input_str=[input_str,')'];
          this.id_ = eval(input_str);
    else
        error('Invalid number of inputs') 
    end
  end

  function delete(this)
  %DELETE Destructor.
    MParT_('ParameterizedFunction_delete', this.id_);
  end

  function SetCoeffs(this,coeffs)
    MParT_('ParameterizedFunction_SetCoeffs',this.id_,coeffs(:));
  end

  function result = Coeffs(this)
    result = MParT_('ParameterizedFunction_Coeffs',this.id_);
  end

  function result = CoeffMap(this)
    result = MParT_('ParameterizedFunction_CoeffMap',this.id_);
  end

  function result = numCoeffs(this)
    result = MParT_('ParameterizedFunction_numCoeffs',this.id_);
  end

  function result = Evaluate(this,pts)
    result = zeros(this.outputDim, size(pts,2));
    MParT_('ParameterizedFunction_Evaluate',this.id_,pts,result);
  end

  function result = CoeffGrad(this,pts,sens)
    result = zeros(this.numCoeffs, size(pts,2));
    MParT_('ParameterizedFunction_CoeffGrad',this.id_,pts,sens,result);
  end

  function result = get_id(this)
    result = this.id_;
  end

  function result = outputDim(this)
    result = MParT_('ParameterizedFunction_outputDim',this.id_);
  end 

  function result = inputDim(this)
    result = MParT_('ParameterizedFunction_inputDim',this.id_);
  end 

end

end
