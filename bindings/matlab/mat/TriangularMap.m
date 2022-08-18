function map = TriangularMap(listCondMaps)
    listCondMap_ids =[];
    for i=1:length(listCondMaps)
        listCondMap_ids=[listCondMap_ids,listCondMaps(i).get_id()]
    end
    map = ConditionalMap(listCondMap_ids);
end