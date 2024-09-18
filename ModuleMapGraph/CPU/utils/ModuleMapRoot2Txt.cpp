#include <iostream>
#include <string>
#include <boost/program_options.hpp>
#include <common_options>
#include <module_map_triplet>

int main(int argc, char* argv[]){

    boost::program_options::options_description desc;
    common_options my_common_options (desc);
    my_common_options.add_input_options (desc);
    //my_common_options.add_output_options (desc, output_format::Root);
    my_common_options.add_graph_options<data_type1> (desc);   

    boost::program_options::variables_map vm = my_common_options.parse (desc, argc, argv);
    if (vm.empty()) return EXIT_FAILURE;

    std::string MM_input_dir = vm["input-dir"].as<std::string>();


    module_map_triplet<data_type1> moduleMap;
    std::string MMfile = MM_input_dir + "/" + vm["input-module-map"].as<std::string>();
    
    std::cout << "Reading input module map in root format: " << MMfile << std::endl;
    moduleMap.read_TTree(MMfile);

    std::string convMMfile = vm["output-module-map"].as<std::string>(); 
    std::cout<< "Saving module map in txt format: " << convMMfile << std::endl;
    moduleMap.save_txt(convMMfile);
}