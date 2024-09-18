/***************************************************************
 * Tracking project library - L2IT
 * Trace reconstruction in LHC
 * copyright © 2023,2024 COLLARD Christophe
 * copyright © 2023,2024 Centre National de la Recherche Scientifique
 * copyright © 2023,2024 Laboratoire des 2 Infinis de Toulouse (L2IT)
 ***************************************************************/

#include <mpi.h>
#include <send>
#include <receive>

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


//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
int master_jobs_distribution (const boost::program_options::variables_map& vm, const int& nb_threads, const std::string& hits_filenames, const std::string& particles_filenames)
//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
{
  std::string events_path = vm["input-dir"].as<std::string>() + "/";
  std::string log_path = vm["log"].as<std::string>() + "/";

  // log all particles events filenames
  std::string cmd_particles = "ls " + events_path + "|grep event |grep particles > " + particles_filenames;
  system (cmd_particles.c_str());

  // log all hits events filenames
  std::string cmd_hits = "ls " + events_path + "|grep event |grep hits > " + hits_filenames;
  system (cmd_hits.c_str());
  cmd_hits = "ls " + events_path + "|grep event |grep truth >> " + hits_filenames;
  system (cmd_hits.c_str());

  // get nb of events
  std::string cmd_info = "wc -l " + particles_filenames + " " + hits_filenames + " > " + log_path + "log.info";
  system (cmd_info.c_str());
  std::ifstream info_file (log_path+"log.info");
  if (info_file.fail()) throw std::invalid_argument ("Cannot open log file log.info");
  int nb_hits, nb_particles;
  info_file >> nb_particles;
  std::cout << "nb of particles files: " << nb_particles << std::endl;
  std::string filename;
  info_file >> filename;
  info_file >> nb_hits;
  std::cout << "nb of hits files: " << nb_hits << std::endl;
  info_file.close ();
  assert (nb_hits == nb_particles);

//  return nb_hits / (nb_threads-1);
  return nb_hits / nb_threads + 1;
}

//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
module_map_triplet<data_type1> master_compute_partial_module_map (boost::program_options::variables_map& vm, const int& nb_threads, const int& my_rank, const int& nb_events, const std::string& hits_filenames, const std::string& particles_filenames)
//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
{
  std::string cmd_parse_files;
  // read staying events (shorter list)
  cmd_parse_files = "sed -n '" + std::to_string ((nb_threads - 1) * nb_events + 1) + ",$ "  + "p' ";
  system ((cmd_parse_files + hits_filenames + " > " + hits_filenames + "." + std::to_string (my_rank)).c_str());
  system ((cmd_parse_files + particles_filenames + " > " + particles_filenames + "." + std::to_string (my_rank)).c_str());
  module_map_triplet<data_type1> my_module_map (vm, hits_filenames + "." + std::to_string (my_rank), particles_filenames + "." + std::to_string (my_rank));
  // temporary save partial Module Map
  if (vm["save-partial-module-maps-on-disk"].as<bool>()) {
    std::cout << "saving MM on thread # " << my_rank << std::endl;
    std::string MM_name = vm["output-module-map"].as<std::string>() + std::to_string(my_rank);
    my_module_map.save_TTree (MM_name);
    my_module_map.save_txt (MM_name);
  }

  return my_module_map;
}

//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
void slaves_compute_partial_module_map (boost::program_options::variables_map& vm, const int& nb_threads, const int& my_rank, const int& nb_events, const std::string& hits_filenames, const std::string& particles_filenames)
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
{
  std::cout << "thread # " << my_rank << " computing " << nb_events << " events" << std::endl;
  std::string cmd_parse_files;
//  if (my_rank == nb_threads-1)
//      cmd_parse_files = "sed -n '" + std::to_string ((my_rank-1) * nb_events + 1) + ",$ "  + "p' ";
//  if (!my_rank)
//      cmd_parse_files = "sed -n '" + std::to_string ((nb_threads-1) * nb_events + 1) + ",$ "  + "p' ";
//  else
  cmd_parse_files = "sed -n '" + std::to_string ((my_rank-1) * nb_events + 1) + "," + std::to_string (my_rank * nb_events) + "p' ";
  system ((cmd_parse_files + hits_filenames + " > " + hits_filenames + "." + std::to_string (my_rank)).c_str());
  system ((cmd_parse_files + particles_filenames + " > " + particles_filenames + "." + std::to_string (my_rank)).c_str());
  module_map_triplet<data_type1> my_module_map (vm, hits_filenames + "." + std::to_string (my_rank), particles_filenames + "." + std::to_string (my_rank));
  send (my_module_map, 0, my_rank);
  // temporary save partial Module Map
  if (vm["save-partial-module-maps-on-disk"].as<bool>()) {
    std::cout << "saving MM on thread # " << my_rank << std::endl;
    std::string MM_name = vm["output-module-map"].as<std::string>() + std::to_string(my_rank);
    my_module_map.save_TTree (MM_name);
    my_module_map.save_txt (MM_name);
  }
}

//---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
void master_module_map_reconstruction (boost::program_options::variables_map& vm, const int& nb_threads, const int& my_rank, const int& nb_events, const std::string& hits_filenames, const std::string& particles_filenames)
//---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
{
  module_map_triplet<data_type1> FullModuleMap = master_compute_partial_module_map (vm, nb_threads, my_rank, nb_events, hits_filenames, particles_filenames);

  for (int thread=1; thread<nb_threads; thread++) {
    module_map_triplet<data_type1> PartialModuleMap;
    receive (PartialModuleMap, thread, thread);
    FullModuleMap.merge (PartialModuleMap);
  }

  FullModuleMap.save_TTree (vm["output-module-map"].as<std::string>());
  FullModuleMap.save_txt (vm["output-module-map"].as<std::string>());
}

//-------------------------------
int main (int argc, char* argv[])
//-------------------------------
{
  MPI::Init (argc, argv);
  int nb_threads = MPI::COMM_WORLD.Get_size();
  int my_rank = MPI::COMM_WORLD.Get_rank();

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

  std::string log_path = vm["log"].as<std::string>();
  if (log_path != "") {
    if (!my_rank) system (("mkdir " + log_path).c_str());
    log_path += "/";
  }
  std::string hits_filenames = log_path + "events-hits.txt";
  std::string particles_filenames = log_path + "events-particles.txt";

  if (!my_rank) {
    start_main = start = clock();
    std::cout << "log = " << log_path << std::endl;
    nb_events = master_jobs_distribution (vm, nb_threads, hits_filenames, particles_filenames);
    end = clock();
    std::cout << red << "cpu time: job scattering = " << (long double)(end-start)/CLOCKS_PER_SEC << reset;
  }

  MPI::COMM_WORLD.Bcast (&nb_events, 1, MPI::INT, 0);

  if (my_rank) {
    start = clock();
    slaves_compute_partial_module_map (vm, nb_threads, my_rank, nb_events, hits_filenames, particles_filenames);
    end = clock();
    std::cout << magenta << "cpu time on thread #" << my_rank << ": module map construction = " << (long double)(end-start)/CLOCKS_PER_SEC << reset;
  }

  if (!my_rank) {
    start = clock();
    master_module_map_reconstruction (vm, nb_threads, my_rank, nb_events, hits_filenames, particles_filenames);
    end = clock();
    std::cout << blue << "cpu time: module map merging = " << (long double)(end-start)/CLOCKS_PER_SEC << reset;
    end = clock();
    std::cout << "module map build cpu time : " << (long double)(end-start_main)/CLOCKS_PER_SEC << std::endl;
    int time = (long double)(end-start_main)/CLOCKS_PER_SEC;
    int min = time / 60;
    int sec = time - 60 * min;
    std::cout << green << "module map build cpu time : " << min << "'" << sec << "\"" << reset;
  }

  MPI::Finalize ();
}
