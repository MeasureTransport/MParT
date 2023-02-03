function map = AdaptiveTransportMap(mset0, objective, atm_options)
    mexOptions = atm_options.getMexOptions;
    input_str=['map = MParT_(',char(39),'ConditionalMap_AdaptiveTransportMap',char(39),',mset0.get_id(),objective.get_id()'];
    for o=1:length(mexOptions)-1
        input_o=[',mexOptions{',num2str(o),'}'];
        input_str=[input_str,input_o];
    end
    maxDegrees = mexOptions{24}.get_id()
    input_str=[input_str,'maxDegrees)'];
    eval(input_str);
end