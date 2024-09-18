/***************************************************************
 * Tracking project library - L2IT
 * Trace reconstruction in LHC
 * copyright © 2022,2023 ROUGIER Charline
 * copyright © 2022,2023,2024 COLLARD Christophe
 * copyright © 2022,2023,2024 Centre National de la Recherche Scientifique
 * copyright © 2022,2023,2024 Laboratoire des 2 Infinis de Toulouse (L2IT)
 ***************************************************************/

#ifndef __cplusplus
#error Must use C++ for the type module map triplet
#endif

#include <time.h>
#include <string>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <map>
#include <TFile.h>
#include <TTree.h>
#include <boost/program_options.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/regex.hpp>
#include <hits>
#include <TTree_hits>
#include <TTree_particles>
#include <hits_infos>
#include <geometric_cuts>
#include <module_doublet>
#include <module_triplet>
#include <module_map_triplet>
#include <edge>
#include <iterator>

//==============================================
template <class T> class CUDA_module_map_triplet
//==============================================
{
  private:
    //---------
    // CPU data
    //---------
    int _nb_triplets;

    //geometric cuts for doublets
    std::vector<T> _module_z0_min, _module_z0_max, _module_dphi_min, _module_dphi_max, _module_phi_slope_min, _module_phi_slope_max, _module_deta_min, _module_deta_max;
    std::vector<int> _MD12_map, _MD23_map;

    // geometric cuts for triplets
    CUDA_module_map_doublet<T> _MD12, _MD23;
    std::vector<T> _diff_dydx_min, _diff_dydx_max, _diff_dzdr_min, _diff_dzdr_max;
    std::vector<unsigned int> _occurence;

    std::vector<int> doublets_map;
    unsigned* map_addresses;

    void allocate_doublets ();
    void allocate_triplets ();

    //---------
    // GPU data
    //---------
    T *_cuda_diff_dydx_min, *_cuda_diff_dydx_max, *_cuda_diff_dzdr_min, *_cuda_diff_dzdr_max;
    int *_cuda_MD12_map, *_cuda_MD23_map;

  public:
    CUDA_module_map_triplet () {_nb_triplets=0;}
    CUDA_module_map_triplet (module_map_triplet<T>&);
    ~CUDA_module_map_triplet ();

//    inline int nb_doublets () const {return _nb_doublets;}
//    inline int nb_triplets () const {return _nb_triplets;}
    inline int size () const {return _nb_triplets;}
    inline int* cuda_module12_1 () {return _MD12.cuda_module1();}
    inline int* cuda_module12_2 () {return _MD12.cuda_module2();}
    inline int* cuda_module23_1 () {return _MD23.cuda_module1();}
    inline int* cuda_module23_2 () {return _MD23.cuda_module2();}
    inline int* cuda_module12_map () {return _cuda_MD12_map;}
    inline int* cuda_module23_map () {return _cuda_MD23_map;}
    inline T* cuda_diff_dydx_min () {return _cuda_diff_dydx_min;}
    inline T* cuda_diff_dydx_max () {return _cuda_diff_dydx_max;}
    inline T* cuda_diff_dzdr_min () {return _cuda_diff_dzdr_min;}
    inline T* cuda_diff_dzdr_max () {return _cuda_diff_dzdr_max;}
    inline CUDA_module_map_doublet<T>& module12 () {return _MD12;}
    inline CUDA_module_map_doublet<T>& module23 () {return _MD23;}
    void HostToDevice ();
};
