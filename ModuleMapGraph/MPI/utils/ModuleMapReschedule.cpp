/***************************************************************
 * Tracking project library - L2IT
 * Trace reconstruction in LHC
 * copyright © 2023 COLLARD Christophe
 * copyright © 2023 Centre National de la Recherche Scientifique
 * copyright © 2023 Laboratoire des 2 Infinis de Toulouse (L2IT)
 ***************************************************************/

#include <parameters>
#include <time.h>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <boost/program_options.hpp>
#include <common_options>

//===============================
int main (int argc, char* argv[])
//===============================
{
  clock_t start, end;

  boost::program_options::options_description desc;

  common_options my_common_options (desc);

  my_common_options.add_sequencer_options (desc);
  my_common_options.add_input_options (desc);
  my_common_options.add_graph_options<data_type1> (desc);

  boost::program_options::variables_map vm = my_common_options.parse (desc, argc, argv);
  if (vm.empty()) return EXIT_FAILURE;

  std::string events_path = vm["input-dir"].as<std::string>() + "/";

  int restart_step = vm["restart-from"].as<int>();
  int end_step = vm["end-at"].as<int>();
  assert (restart_step >= 0  &&  (!end_step || end_step >= restart_step));

  if (!restart_step) {
    std::string cmd_dir_root_triplets = "ls " + events_path + "*.triplets.root > grid.reschedule.files.txt";
    system (cmd_dir_root_triplets.c_str());
    std::string cut_path = "sed -i -e 's|" + events_path + vm["output-module-map"].as<std::string>() + ".||' grid.reschedule.files.txt";
    system (cut_path.c_str());
    system ("cut -f 1 -d '.' grid.reschedule.files.txt > grid.reschedule.filenames.txt");
    std::string cmd_grep = "grep -f grid.reschedule.filenames.txt -v " + events_path + "grid.files.txt > " + events_path + "grid.reschedule.files.txt";
    std::cout << cmd_grep << std::endl;
    system (cmd_grep.c_str());
//    system ("sort -d grid.filenames.txt > grid.files.txt");
//    system (("cut -d . -f 1 grid.files.txt > " + events_path + "grid.files.txt").c_str());
//    system ("rm grid.filenames.txt");
  }

  std::ifstream event_files ((events_path+"grid.reschedule.files.txt").c_str());
  if (event_files.fail()) throw std::invalid_argument ("Cannot open log file grid.filenames.txt");

  int step = 0;
  std::string name;
  for (event_files >> name; !event_files.eof(); event_files >> name) {
    std::string log_dir = events_path + "log." + name;

    if (++step < restart_step && restart_step) continue;

    std::cout << "processing file " << name << std::endl;
    struct stat info;
    if (stat (log_dir.c_str(), &info) != 0) {
      system (("mkdir " + log_dir).c_str());
      std::string cmd_grid_cp = "rfcp " + vm["grid-input-dir"].as<std::string>() + "/" + name + ".tar " + log_dir;
      system (cmd_grid_cp.c_str());
      std::string cmd_untar = "tar xvf " + log_dir + "/" + name + ".tar" + " -C " + log_dir;
      system (cmd_untar.c_str());
      std::string cmd_rm = "rm " + log_dir + "/" + name + ".tar";
      system (cmd_rm.c_str());
    }

    std::string cmd_sbatch = "sbatch -o " + log_dir + "/ModuleMapCreator.mpi.log " + "slurm/ModuleMapCreator.scheduler.slurm " + log_dir + " " + log_dir + "/" + name + "_geo " + log_dir + " " + name;
    system (cmd_sbatch.c_str());
    std::cout << "job submission: " << cmd_sbatch << std::endl;
    if (end_step  &&  step >= end_step) break;
  }
}
