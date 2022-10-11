function map = ComposedMap(listTriMaps)
    listTriMaps_ids =[];
    for i=1:length(listCondMaps)
        listTriMaps_ids=[listTriMaps_ids,listTriMaps(i).get_id()];
    end
    map = ConditionalMap(listTriMaps_ids,"compose");
end