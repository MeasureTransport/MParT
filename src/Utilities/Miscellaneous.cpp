#include "MParT/Utilities/Miscellaneous.h"

using namespace mpart;

std::string mpart::GetOption(std::unordered_map<std::string,std::string> const& map, 
                             std::string                                 const& key, 
                             std::string                                 const& defaultValue)
{

    // Extract the polynomial type
    std::string output;
    auto it = map.find(key);
    if(it==map.end()){
        output = defaultValue;
    }else{
        output = map.at(key);
    }
    return output;
}