classdef ConditionalMap < handle
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

  function this = ConditionalMap(varargin)

    if(nargin==2)
        mset = varargin{1};
        mapOptions = varargin{2};
        mexOptions = mapOptions.getMexOptions;

        input_str=['MParT_(',char(39),'ConditionalMap_newMap',char(39),',mset.get_id()'];
        for o=1:length(mexOptions)
            input_o=[',mexOptions{',num2str(o),'}'];
            input_str=[input_str,input_o];
        end
        input_str=[input_str,')'];
        
        this.id_ = eval(input_str);

    elseif(nargin==4)
      inputDim = varargin{1};
      outputDim = varargin{2};
      totalOrder = varargin{3};
      opts = varargin{4};
      
      mexOptions = opts.getMexOptions;

      input_str=['MParT_(',char(39),'ConditionalMap_newTotalTriMap',char(39),',inputDim,outputDim,totalOrder'];
      for o=1:length(mexOptions)
        input_o=[',mexOptions{',num2str(o),'}'];
        input_str=[input_str,input_o];
      end
      input_str=[input_str,')'];
      this.id_ = eval(input_str);

    elseif(nargin==1)
         MParT_('ConditionalMap_newTriMap', varargin{1});

    else
        error('Invalid number of inputs') 
    end
  end

  function delete(this)
  %DELETE Destructor.
    MParT_('ConditionalMap_deleteMap', this.id_);
  end

  function SetCoeffs(this,coeffs)
    MParT_('ConditionalMap_SetCoeffs',this.id_,coeffs(:));
  end

  function result = Coeffs(this)
    result = MParT_('ConditionalMap_Coeffs',this.id_);
  end

  function result = numCoeffs(this)
    result = MParT_('ConditionalMap_numCoeffs',this.id_);
  end

  function result = Evaluate(this,pts)
    result = zeros(this.outputDim, size(pts,2));
    MParT_('ConditionalMap_Evaluate',this.id_,pts,result);
  end

  function result = LogDeterminant(this,pts)
    result = zeros(size(pts,2),1);
    MParT_('ConditionalMap_LogDeterminant',this.id_,pts,result);
  end

  function result = Inverse(this,x1,r)
    result = zeros(this.outputDim, size(r,2));
    MParT_('ConditionalMap_Inverse',this.id_,x1,r,result);
  end

  function result = CoeffGrad(this,pts,sens)
    result = zeros(this.numCoeffs, size(pts,2));
    MParT_('ConditionalMap_CoeffGrad',this.id_,pts,sens,result);
  end

  function result = LogDeterminantCoeffGrad(this,pts)
    result = zeros(this.numCoeffs, size(pts,2));
    MParT_('ConditionalMap_LogDeterminantCoeffGrad',this.id_,pts,result);
  end

  function result = get_id(this)
    result = this.id_;
  end

  function result = outputDim(this)
    result = MParT_('ConditionalMap_outputDim',this.id_);
  end 

  function result = inputDim(this)
    result = MParT_('ConditionalMap_inputDim',this.id_);
  end 

end

end
