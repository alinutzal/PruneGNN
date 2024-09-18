/***************************************************************
 * Tracking project library - L2IT
 * Trace reconstruction in LHC
 * copyright © 2024 COLLARD Christophe
 * copyright © 2024 Centre National de la Recherche Scientifique
 * copyright © 2024 Laboratoire des 2 Infinis de Toulouse (L2IT)
 ***************************************************************/

#ifndef __cplusplus
#error Must use C++ for the type TTree hits
#endif

#include <parameters>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <utility>
#include <set>
//#include <algorithm>
//#include <execution>
#include <vector>
#include <TFile.h>
#include <TTree.h>
#include <boost/type_index.hpp>
#include <TTree_hits>
#include <CUDA_module_map_doublet>
#include <hit>
#include <hits>


//=============================================================
template <class T> class CUDA_TTree_hits : public TTree_hits<T>
//=============================================================
{
  using TTree_hits<T>::_size;
  using TTree_hits<T>::_position;
  using TTree_hits<T>::_x, TTree_hits<T>::_y, TTree_hits<T>::_z;
  using TTree_hits<T>::_true_graph, TTree_hits<T>::_extra_features;
  using TTree_hits<T>::_hit_id;
//  using TTree_hits<T>::_R, TTree_hits<T>::_eta, TTree_hits<T>::_phi;
  using TTree_hits<T>::_moduleID_TTreeHits_map;

  private:
    //---------
    // CPU data
    //---------
    std::vector<int> _hit_indice; // indices for pts: _indice_first+1 -> indice_last
    std::vector<int> _vertices;
    std::vector<T> _R, _eta, _phi;

    //---------
    // GPU data
    //---------
    T *_cuda_x, *_cuda_y, *_cuda_z;
    T *_cuda_R, *_cuda_eta, *_cuda_phi;
    uint64_t *_cuda_hit_id;
    int *_cuda_hit_indice, *_cuda_vertices;

  public:
//    CUDA_TTree_hits () {}
    CUDA_TTree_hits (bool true_graph, bool extra_features): TTree_hits<T> (true_graph, extra_features) {}
    CUDA_TTree_hits (TTree_hits<T>&, const std::multimap<uint64_t,int>&);
//    TTree_hits (hits<T>&);  // cast conversion to hits
    ~CUDA_TTree_hits ();

//    inline void build_R () {std::transform (std::execution::par_unseq, std::begin(_x), std::end(_x), std::begin(_y), std::begin(_R), [](data_type2 xi, data_type2 yi) {return sqrt (xi * xi + yi * yi);});}

    void initialize (const std::multimap<uint64_t,int>&);
    inline T R (int i) {return _R[i];}
    inline T eta (int i) const {return _eta[i];}
    inline T phi (int i) const {return _phi[i];}

    inline T R () const {return _R[_position];}
    inline T eta () const {return _eta[_position];}
    inline T phi () const {return _phi[_position];}
    inline std::vector<int>& vertices () {return _vertices;}
    void DeviceToHost ();

    void HostToDevice();
    inline uint64_t* cuda_hit_id () {return _cuda_hit_id;}
    inline T* cuda_x () {return _cuda_x;}
    inline T* cuda_y () {return _cuda_y;}
    inline T* cuda_z () {return _cuda_z;}
    inline T* cuda_R () {return _cuda_R;}
    inline T* cuda_eta () {return _cuda_eta;}
    inline T* cuda_phi () {return _cuda_phi;}
    inline int* cuda_hit_indice () {return _cuda_hit_indice;}
    inline int* cuda_vertices () {return _cuda_vertices;}
};
