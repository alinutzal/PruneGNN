#include <time.h>
#include <iostream>
#include <string>
#include <boost/program_options.hpp>
#include <common_options>
#include <module_map_triplet>


//-------------------------------
int main (int argc, char* argv[])
//-------------------------------
{
  clock_t start, end;
  start = clock ();

  boost::program_options::options_description desc;
  common_options my_common_options (desc);
  my_common_options.add_input_options (desc);
  //my_common_options.add_output_options (desc, output_format::Root);
  my_common_options.add_graph_options<data_type1> (desc);   

  boost::program_options::variables_map vm = my_common_options.parse (desc, argc, argv);
  if (vm.empty()) return EXIT_FAILURE;

  std::string MM_input_dir = vm["input-dir"].as<std::string>();

  module_map_triplet<data_type1> FullModuleMap;
  std::string rawMMfile = MM_input_dir + "/" + vm["input-module-map"].as<std::string>();

  std::cout << "Reading input module map: " << rawMMfile << std::endl;
  FullModuleMap.read_TTree(rawMMfile);

  std::cout<< "Removing cycles in Module Map" << std::endl;
  FullModuleMap.remove_unique_occurence();
  FullModuleMap.remove_cycles();

  std::cout<< "Saving cleaned Module Map" << std::endl;
  std::string cleanedMMfile = MM_input_dir + vm["output-module-map"].as<std::string>(); 
  FullModuleMap.save_TTree(cleanedMMfile);
  FullModuleMap.save_txt(cleanedMMfile);

  end = clock();

  std::cout << "graph build cpu time : " << (long double)(end-start)/CLOCKS_PER_SEC << std::endl;
  std::cout << "end read file" << std::endl;
  int time = (long double)(end-start)/CLOCKS_PER_SEC;
  int min = time / 60;
  int sec = time - 60 * min;
  std::cout << green << "graph build cpu time : " << min << "'" << sec << "\"" << reset;

  return 0;
}
