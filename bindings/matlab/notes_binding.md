# Filling _mex.cpp files

# Filling .m files

# getConst
```
const MultiIndexSet& mset = Session<MultiIndexSet>::getConst(input.get(0))
output.set(0, mset.Frontier())
```


# get
```
MultiIndexSet *mset = Session<MultiIndexSet>::get(input.get(0))
output.set(0, mset->ForciblyExpand(activeIndex))
```


# id_
