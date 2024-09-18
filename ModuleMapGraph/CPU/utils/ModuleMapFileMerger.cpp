/***************************************************************
 * Tracking project library - L2IT
 * Trace reconstruction in LHC
 * copyright © 2023,2024 COLLARD Christophe
 * copyright © 2023,2024 Centre National de la Recherche Scientifique
 * copyright © 2023,2024 Laboratoire des 2 Infinis de Toulouse (L2IT)
 ***************************************************************/

#include <time.h>
#include <iostream>
#include <string>
#include <boost/program_options.hpp>
#include <common_options>
#include <hits>
#include <TTree_hits>
#include <particles>
#include <TTree_particles>
#include <module_map_triplet>
#include <colors>
#include <parameters>


//-------------------------------
int main (int argc, char* argv[])
//-------------------------------
{
  clock_t start_main, start, end;
  int nb_events = 0;

  boost::program_options::options_description desc;

  common_options my_common_options (desc);

  my_common_options.add_sequencer_options (desc);
  // Options::addGeometryOptions (desc);
  //Options::addMaterialOptions (desc);
  my_common_options.add_input_options (desc);
  //  Options::addGraphCreationOptions (desc);
  my_common_options.add_output_options (desc, output_format::Root);
  my_common_options.add_graph_options<data_type1> (desc);

  //  my_common_options.read_sequencer_config (vm)

  // detector.addOptions(desc);

  boost::program_options::variables_map vm = my_common_options.parse (desc, argc, argv);
  if (vm.empty()) return EXIT_FAILURE;

  std::string MM_input_dir = vm["input-dir"].as<std::string>();

  // log all Module Map filenames
  std::string log_MM_filenames = "module-maps.txt";
  std::string cmd_list_MM = "ls --ignore=*.triplets.root.txt " + MM_input_dir + "|grep triplets.root | sed -n s/.triplets.root$//p > " + MM_input_dir + "/" + log_MM_filenames;
  system (cmd_list_MM.c_str());
//  std::string cmd_info = "wc -l module-maps.txt > log.MM.info";
//  system (cmd_info.c_str());
  // open log file
  std::ifstream MM_file (MM_input_dir + "/" + log_MM_filenames);
  if (MM_file.fail()) throw std::invalid_argument ("Cannot open file " + log_MM_filenames);

  int restart_step = vm["restart-from"].as<int>();
  int end_step = vm["end-at"].as<int>();

  module_map_triplet<data_type1> FullModuleMap;
  std::string MM_filename;

  int counter = 1;
  for (MM_file >> MM_filename; !MM_file.eof(); MM_file >> MM_filename, counter++) {
    if (restart_step)
      if (counter < restart_step || counter > end_step) continue;
    module_map_triplet<data_type1> PartialModuleMap;
    PartialModuleMap.read_TTree (MM_input_dir + "/" + MM_filename);
    FullModuleMap.merge (PartialModuleMap);
  }

//  std::string output_filename = MM_input_dir + "/" + vm["output-module-map"].as<std::string>();
  std::string output_filename = vm["output-module-map"].as<std::string>();
  if (restart_step) output_filename += "-" + std::to_string(restart_step) + "-" + std::to_string(end_step);

  FullModuleMap.save_TTree (output_filename);
  FullModuleMap.save_txt (output_filename);

//  FullModuleMap.read_TTree (MM_input_dir + "/" + vm["output-module-map"].as<std::string>());
//  FullModuleMap.sort();
//  FullModuleMap.save_TTree (MM_input_dir + "/Sorted." + vm["output-module-map"].as<std::string>());
//  FullModuleMap.save_txt (MM_input_dir + "/Sorted." + vm["output-module-map"].as<std::string>());
}
