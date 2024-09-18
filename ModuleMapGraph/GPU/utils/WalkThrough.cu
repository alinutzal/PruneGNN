/***************************************************************
 * Tracking project library - L2IT
 * Trace reconstruction in LHC
 * copyright © 2024 COLLARD Christophe
 * copyright © 2024 Centre National de la Recherche Scientifique
 * copyright © 2024 Laboratoire des 2 Infinis de Toulouse (L2IT)
 ***************************************************************/

#include <time.h>
#include <iostream>
#include <string>
#include <boost/program_options.hpp>
#include <common_options>
#include <parameters>
#include <CUDA_walkthrough>
#include <colors>
#include <cuda_profiler_api.h>

//-------------------------------
int main (int argc, char* argv[])
//-------------------------------
{
  clock_t start, end;

  start = clock();

  boost::program_options::options_description desc;
  common_options my_common_options (desc);
  my_common_options.add_hardware_options (desc);
  my_common_options.add_sequencer_options (desc);
  // Options::addGeometryOptions (desc);
  //Options::addMaterialOptions (desc);
  my_common_options.add_input_options (desc);
  //  Options::addGraphCreationOptions (desc);
  my_common_options.add_output_options (desc, output_format::Root);
  my_common_options.add_graph_options<data_type1> (desc);

  boost::program_options::variables_map vm = my_common_options.parse (desc, argc, argv);
  if (vm.empty()) return EXIT_FAILURE;

  /////////////////
  // build graph //
  /////////////////

  std::cout << red << "-------------------" << reset;
  std::cout << red << "WalkThrough on GPU" << reset;
  std::cout << red << "-------------------" << reset;

  start = clock();
  CUDA_walkthrough<data_type1> MM_graph (vm);
  end = clock();

  std::cout << "walk through cpu time : " << (long double)(end-start)/CLOCKS_PER_SEC << std::endl;
  std::cout << "end read file" << std::endl;
  int time = (long double)(end-start)/CLOCKS_PER_SEC;
  int min = time / 60;
  int sec = time - 60 * min;
  std::cout << green << "walk through cpu time : " << min << "'" << sec << "\"" << reset;

  return 0;
};
