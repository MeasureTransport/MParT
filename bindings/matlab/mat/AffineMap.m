function map = AffineMap(varargin)
    if(nargin == 1)
        % this not very safe (matlab vector is a matrix and vectors can be vertical or horizontal)
        if isvector(varargin{1})
            map = ConditionalMap(varargin{1},"b");
        else
            map = ConditionalMap(varargin{1},"A");
        end
    elseif(nargin == 2)
        map = ConditionalMap(varargin{1},varargin{2},"Ab");
    else
    error("Wrong number of input arguments");
    end
    
end