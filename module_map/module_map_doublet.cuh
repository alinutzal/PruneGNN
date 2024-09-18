/***************************************************************
 * Tracking project library - L2IT
 * Trace reconstruction in LHC
 * copyright © 2024 COLLARD Christophe
 * copyright © 2024 Centre National de la Recherche Scientifique
 * copyright © 2024 Laboratoire des 2 Infinis de Toulouse (L2IT)
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
template <class T> class CUDA_module_map_doublet
//==============================================
{
  private:
    //---------
    // CPU data
    //---------
    int _nb_doublets;

    //geometric cuts for doublets
    std::vector<T> _z0_min, _z0_max, _dphi_min, _dphi_max, _phi_slope_min, _phi_slope_max, _deta_min, _deta_max;

    std::vector<int> _module1, _module2;
    std::multimap<uint64_t, int> _module_map;
    unsigned* map_addresses;

    void allocate_doublets ();

    //---------
    // GPU data
    //---------
    T *_cuda_z0_min, *_cuda_z0_max, *_cuda_dphi_min, *_cuda_dphi_max, *_cuda_phi_slope_min, *_cuda_phi_slope_max, *_cuda_deta_min, *_cuda_deta_max;
    int *_cuda_module1, *_cuda_module2;

  public:
    CUDA_module_map_doublet () {_nb_doublets=0;}
    CUDA_module_map_doublet (module_map_triplet<T>&);
    ~CUDA_module_map_doublet ();

    void build_doublets (module_map_triplet<T>&, std::string);
    inline int size () const {return _nb_doublets;}
    inline const std::multimap <uint64_t, int>& module_map () const {return _module_map;}
    inline std::vector<int> module1 () {return _module1;}
    inline std::vector<int> module2 () {return _module2;}

    inline int* cuda_module1 () {return _cuda_module1;}
    inline int* cuda_module2 () {return _cuda_module2;}
    inline T* cuda_z0_min () {return _cuda_z0_min;}
    inline T* cuda_z0_max () {return _cuda_z0_max;}
    inline T* cuda_dphi_min () {return _cuda_dphi_min;}
    inline T* cuda_dphi_max () {return _cuda_dphi_max;}
    inline T* cuda_phi_slope_min () {return _cuda_phi_slope_min;}
    inline T* cuda_phi_slope_max () {return _cuda_phi_slope_max;}
    inline T* cuda_deta_min () {return _cuda_deta_min;}
    inline T* cuda_deta_max () {return _cuda_deta_max;}
    void HostToDevice ();
};
