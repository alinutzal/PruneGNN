/***************************************************************
 * Tracking project library - L2IT
 * Trace reconstruction in LHC
 * copyright © 2024 COLLARD Christophe
 * copyright © 2024 Centre National de la Recherche Scientifique
 * copyright © 2024 Laboratoire des 2 Infinis de Toulouse (L2IT)
 ***************************************************************/

#ifndef __cplusplus
#error Must use C++ for the type graph creator
#endif

#include <colors>
#include <string>
#include <iostream>
#include <TMath.h>
#include <filesystem>
#include <boost/program_options.hpp>
#include <boost/regex.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <module_map_triplet>
#include <CUDA_module_map_doublet>
#include <CUDA_module_map_triplet>
#include <hits>
#include <particles>
#include <TTree_hits>
#include <CUDA_TTree_hits>
#include <TTree_particles>
#include <hits_infos>
#include <graph>
//#include <graph_true>
#include <strip_hit_pair>
#include <strip_module_DB>
#include <CUDA_quick_sort>
#include <CUDA_prescan>
#include <CUDA_stream_compaction>
#include <CUDA_geometric_cuts>


//=========================================
template <class T> class CUDA_graph_creator
//=========================================
{
  private:
    // CPU data
    std::string _events_dir, _output_dir, _module_map_dir;
    bool _true_graph, _save_graph_graphml, _save_graph_npz, _save_graph_pyg, _save_graph_csv;
    bool _strip_hit_pair, _extra_features;
    T _min_pt, _max_pt;
    long unsigned int _min_N_hits;
    bool _phi_slice;
    T _phi_slice_cut1;
    T _phi_slice_cut2;
    bool _eta_region;
    T _eta_cut1;
    T _eta_cut2;

    // GPU data
    int _blocks;

    strip_module_DB<data_type2> _strip_module_DB;
    CUDA_module_map_doublet<T> _cuda_module_map_doublet;
    CUDA_module_map_triplet<T> _cuda_module_map_triplet;


  public:
    CUDA_graph_creator (boost::program_options::variables_map&);
    ~CUDA_graph_creator () {}

    void build (std::string, std::string);
    void save_csv (std::string = "");
};
