/***************************************************************
 * Tracking project library - L2IT
 * Trace reconstruction in LHC
 * copyright © 2022,2023 ROUGIER Charline
 * copyright © 2022,2023 COLLARD Christophe
 * copyright © 2022,2023 Centre National de la Recherche Scientifique
 * copyright © 2022,2023 Laboratoire des 2 Infinis de Toulouse (L2IT)
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
#include <colors>
#include <parameters>

//-------------------------------------------------------------------------------------------------------------------------------------------------
void convert_file (const std::string& input_path, const std::string output_path, const std::string& hits_filename, std::string& particles_filename)
//-------------------------------------------------------------------------------------------------------------------------------------------------
{
  //-------------
  // get event ID
  //-------------
  std::string event_id = boost::regex_replace (hits_filename, boost::regex("[^0-9]*([0-9]+).*"), std::string("\\1"));
  if (event_id != boost::regex_replace (particles_filename, boost::regex("[^0-9]*([0-9]+).*"), std::string("\\1")))
    throw std::invalid_argument ("hits and particules file are not part of the same event");

  //-------------------------
  // read the whole hits file
  //-------------------------
  hits<data_type1> hit_pt ;
  hit_pt.read_csv_once (input_path, hits_filename);
  TTree_hits<data_type1> ht_root (hit_pt);
  std::string hits_rootfilename = hits_filename;
  hits_rootfilename.erase (hits_rootfilename.size() - 9);
  hits_rootfilename += "hits.root";
  ht_root.save (output_path + hits_rootfilename);

  //------------------------------
  // read the whole particles file
  //------------------------------
  particles<data_type1> part;
  part.read_csv_once (input_path, particles_filename);
  TTree_particles<data_type1> p_root (part);
  std::string particles_rootfilename = particles_filename;
  particles_rootfilename.erase (particles_rootfilename.size() - 4);
  particles_rootfilename += ".root";
  p_root.save (output_path + particles_rootfilename);
}

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
  std::string events_output_path = vm["output-dir"].as<std::string>() + "/";

  // log all particles events filenames
  std::string log_particles_filenames = "events-particles.txt";
  std::string cmd_particles = "ls " + events_path + "|grep event |grep particles.csv > " + log_particles_filenames;
  system (cmd_particles.c_str());
  // open log file
  std::ifstream p_file (log_particles_filenames);
  if (p_file.fail()) throw std::invalid_argument ("Cannot open file " + log_particles_filenames);

  // log all hits events filenames
  std::string log_hits_filenames = "events-hits.txt";
  std::string cmd_hits = "ls " + events_path + "|grep event |grep truth.csv > " + log_hits_filenames;
  system (cmd_hits.c_str());
  // open log file  
  std::ifstream h_file (log_hits_filenames);
  if (h_file.fail()) throw std::invalid_argument ("Cannot open file " + log_hits_filenames);

  //----------------------------------------------------
  // parallel computation on events OR loop on filenames
  //----------------------------------------------------
  std::string particles_filename, hits_filename;

  for (; !p_file.eof() && !h_file.eof();) {
    p_file >> particles_filename;
    h_file >> hits_filename;

//    std::cout << "input  directory: " << events_path << std::endl;
//    std::cout << "output directory: " << events_output_path << std::endl;

    std::cout << "hit filename: " << hits_filename << std::endl;
    std::cout << "particles filename: " << particles_filename << std::endl;

    std::string hits_location = events_path + hits_filename;
    std::string particles_location = events_path + particles_filename;

    convert_file (events_path, events_output_path, hits_filename, particles_filename);
  }

  // do this 2 lines by master thread
  p_file.close();
  h_file.close();

  return 0;
}
