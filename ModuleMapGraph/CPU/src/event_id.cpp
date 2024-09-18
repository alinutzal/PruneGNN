#include "event_id.hpp"
#include <boost/regex.hpp>

std::string extract_event_id(const std::string& filename){
  
  // Remove first version number string like vX and vXY (if more than two digits it will break)
  std::string filename_no_version = boost::regex_replace(filename, boost::regex("v[0-9]{0,2}"), std::string(""));

  return boost::regex_replace(filename_no_version, boost::regex("[^0-9]*([0-9]+).*"), std::string("\\1"));
}
