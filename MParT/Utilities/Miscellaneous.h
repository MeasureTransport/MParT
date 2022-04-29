#ifndef MPART_MISCELLANEOUS_H
#define MPART_MISCELLANEOUS_H

#include <unordered_map>
#include <string>

namespace mpart{

    /** Tries to read an options from a std::map.  If the key does not exist, the specified default value is returned. */
    std::string GetOption(std::unordered_map<std::string,std::string> const& map, 
                          std::string                                 const& key, 
                          std::string                                 const& defaultValue);
}

#endif 