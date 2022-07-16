classdef Normal1D
    %UNTITLED3 Summary of this class goes here
    %   Detailed explanation goes here

    properties
        mean=0;
        std=1;
    end

    methods
        function result = LogPdf(obj,x)
            %UNTITLED3 Construct an instance of this class
            %   Detailed explanation goes here
            result = -log(obj.std*sqrt(2*pi)) - 0.5*((x-obj.mean)/obj.std).^2;
        end

        function result = GradLogPdf(obj,x)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            result = - ((x-obj.mean)/obj.std);
        end
    end
end