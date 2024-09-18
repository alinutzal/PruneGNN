/***************************************************************
 * Tracking project library - L2IT
 * Trace reconstruction in LHC
 * copyright © 2022,2023 ROUGIER Charline
 * copyright © 2022,2023,2024 COLLARD Christophe
 * copyright © 2022,2023,2024 Centre National de la Recherche Scientifique
 * copyright © 2022,2023,2024 Laboratoire des 2 Infinis de Toulouse (L2IT)
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
  clock_t start, end;
  clock_t start_main, end_main;

  start_main = start = clock();

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

  std::string events_path = vm["input-dir"].as<std::string>() + "/";
  std::string event_filename_pattern = vm["input-filename-pattern"].as<std::string> ();

  // log all particles events filenames
  std::string log_particles_filenames = "events-particles.txt";
  std::string cmd_particles = "ls " + events_path + "|grep " + event_filename_pattern + " |grep particles.csv > " + log_particles_filenames;
  system (cmd_particles.c_str());
  // open log file
  std::ifstream p_file (log_particles_filenames);
  if (p_file.fail()) throw std::invalid_argument ("Cannot open file " + log_particles_filenames);

  // log all hits events filenames
  std::string log_hits_filenames = "events-hits.txt";
  std::string cmd_hits = "ls " + events_path + "|grep " + event_filename_pattern + " |grep hits.csv > " + log_hits_filenames;
  system (cmd_hits.c_str());
  cmd_hits = "ls " + events_path + "|grep " + event_filename_pattern + " |grep truth.csv >> " + log_hits_filenames;
  system (cmd_hits.c_str());
  // open log file  
  std::ifstream h_file (log_hits_filenames);
  if (h_file.fail()) throw std::invalid_argument ("Cannot open file " + log_hits_filenames);

  //----------------------------------------------------
  // parallel computation on events OR loop on filenames
  //----------------------------------------------------
  std::string particles_filename, hits_filename;
  p_file >> particles_filename;
  h_file >> hits_filename;

  // do this 2 lines by master thread
  p_file.close();
  h_file.close();

  //////////////////////
  // build Module Map //
  //////////////////////

  start = clock();
  std::string root_particles_filenames = "root-events-particles.txt";
  std::string cmd_root_particles = "ls " + events_path + "|grep event |grep particles.root > " + root_particles_filenames;
  system (cmd_root_particles.c_str());
  cmd_root_particles = "ls " + events_path + "|grep event |grep particles.csv >> " + root_particles_filenames;
  system (cmd_root_particles.c_str());
  std::string root_hits_filenames = "root-events-hits.txt";
  std::string cmd_root_hits = "ls " + events_path + "|grep event |grep hits.root > " + root_hits_filenames;
  system (cmd_root_hits.c_str());
  cmd_root_hits = "ls " + events_path + "|grep event |grep hits.csv >> " + root_hits_filenames;
  system (cmd_root_hits.c_str());
  cmd_root_hits = "ls " + events_path + "|grep event |grep truth.csv >> " + root_hits_filenames;
  system (cmd_root_hits.c_str());
  module_map_triplet<data_type1> my_module_map (vm, root_hits_filenames, root_particles_filenames);
  my_module_map.save_TTree (vm["output-module-map"].as<std::string>());
  my_module_map.save_txt (vm["output-module-map"].as<std::string>());

  if( vm["input-module-map"].as<std::string>() != "" ){
    module_map_triplet<data_type1> charline_module_map;
    //charline_module_map.read_TTree ("MMTriplet_3hits_ptCut300MeV_events_woutOverlapSP_particleInfo.root");
    charline_module_map.read_TTree (vm["input-module-map"].as<std::string>());
    if (my_module_map == charline_module_map) std::cout << green << "Module Map Build OK" << reset;
    else std::cout << red << "Module Map Build FAILED" << reset;
  //  module_map_creator<data_type1,data_type2> my_module_map (vm);
  //  my_module_map.build("events/event000000421-hits.root", "events/event000000421-particles.root");
  //  module_map_triplet<data_type1,data_type2> my_module_map (vm, "events/event000000421-hits.root", "events/event000000421-particles.root");
  }
  end = clock();
  std::cout << "module map build cpu time : " << (long double)(end-start)/CLOCKS_PER_SEC << std::endl;
  int time = (long double)(end-start)/CLOCKS_PER_SEC;
  int min = time / 60;
  int sec = time - 60 * min;
  std::cout << green << "module map build cpu time : " << min << "'" << sec << "\"" << reset;

  std::cout << "end of run"<< std::endl;

  return 0;
};
