/***************************************************************
 * Tracking project library - L2IT
 * Trace reconstruction in LHC
 * copyright © 2024 COLLARD Christophe
 * copyright © 2024 Centre National de la Recherche Scientifique
 * copyright © 2024 Laboratoire des 2 Infinis de Toulouse (L2IT)
 ***************************************************************/

#ifndef __cplusplus
#error Must use C++ for the type CUDA walk through
#endif

#include <string>
#include <iostream>
#include <boost/regex.hpp>
#include <CUDA_graph>
#include <CUDA_module_map_doublet>
#include <CUDA_module_map_triplet>
#include <CUDA_quick_sort>
#include <CUDA_prescan>
#include <CUDA_mask>
#include <CUDA_stream_compaction>

//=======================================
template <class T> class CUDA_walkthrough
//=======================================
{
  private:
    int _size;
    std::string _events_dir, _output_dir, _module_map_dir;
    int * _first_vertex;
    int* _edge_range;

    // GPU data
    int _blocks;

    CUDA_module_map_doublet<T> _cuda_module_map_doublet;
    CUDA_module_map_triplet<T> _cuda_module_map_triplet;

    void read_event (const std::string&);

  public:
    CUDA_walkthrough (boost::program_options::variables_map&);
    ~CUDA_walkthrough () {}

    void incoming_vertex (const CUDA_graph<T>&);
};
