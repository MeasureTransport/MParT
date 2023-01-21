function TrainMap(map, objective, train_options)
    mexOptions = train_options.getMexOptions
    input_str=['MParT_(',char(39),'ConditionalMap_TrainMap',char(39),',map.get_id(),objective.get_id(),'];
    for o=1:length(mexOptions)
        input_o=[',mexOptions{',num2str(o),'}'];
        input_str=[input_str,input_o];
    end
    input_str=[input_str,')'];
    eval(input_str);
end