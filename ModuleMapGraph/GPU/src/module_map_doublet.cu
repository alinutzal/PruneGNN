/***************************************************************
 * Tracking project library - L2IT
 * Trace reconstruction in LHC
 * copyright © 2024 COLLARD Christophe
 * copyright © 2024 Centre National de la Recherche Scientifique
 * copyright © 2024 Laboratoire des 2 Infinis de Toulouse (L2IT)
 ***************************************************************/

#include "module_map_doublet.cuh"


//=====Private methods for cuda module map doublet======================================

//---------------------------------------------------
template <class T>
void CUDA_module_map_doublet<T>::allocate_doublets ()
//---------------------------------------------------
{
  std::cout << "memory allocation on GPU" << std::endl;
  cudaMalloc (&_cuda_module1, _nb_doublets * sizeof(int));
  cudaMalloc (&_cuda_module2, _nb_doublets * sizeof(int));
  // GPU memory allocation
  cudaMalloc (&_cuda_z0_min, _nb_doublets * sizeof(T));
  cudaMalloc (&_cuda_z0_max, _nb_doublets * sizeof(T));
  cudaMalloc (&_cuda_dphi_min, _nb_doublets * sizeof(T));
  cudaMalloc (&_cuda_dphi_max, _nb_doublets * sizeof(T));
  cudaMalloc (&_cuda_phi_slope_min, _nb_doublets * sizeof(T));
  cudaMalloc (&_cuda_phi_slope_max, _nb_doublets * sizeof(T));
  cudaMalloc (&_cuda_deta_min, _nb_doublets * sizeof(T));
  cudaMalloc (&_cuda_deta_max, _nb_doublets * sizeof(T));
}

//=====Public methods for cuda module map triplet=======================================


//------------------------------------------------------------------------------
template <class T>
CUDA_module_map_doublet<T>::CUDA_module_map_doublet (module_map_triplet<T>& MMT)
//------------------------------------------------------------------------------
{
  _nb_doublets = MMT.map_doublet().size();
  _module_map = MMT.module_map();

  for (std::pair<std::vector<uint64_t>, int> pairs : MMT.map_pairs ()) {
    _module1.push_back (_module_map.find (pairs.first[0]) -> second);
    _module2.push_back (_module_map.find (pairs.first[1]) -> second);
  }

  for (std::_Rb_tree_iterator<std::pair<const std::vector<uint64_t>, module_doublet<T>>> it2 = MMT.map_doublet().begin(); it2 != MMT.map_doublet().end(); it2++) {
    _z0_min.push_back (it2->second.z0_min());
    _z0_max.push_back (it2->second.z0_max());
    _dphi_min.push_back (it2->second.dphi_min());
    _dphi_max.push_back (it2->second.dphi_max());
    _phi_slope_min.push_back (it2->second.phi_slope_min());
    _phi_slope_max.push_back (it2->second.phi_slope_max());
    _deta_min.push_back (it2->second.deta_min());
    _deta_max.push_back (it2->second.deta_max());
  }
}

//-----------------------------------------------------
template <class T>
CUDA_module_map_doublet<T>::~CUDA_module_map_doublet ()
//-----------------------------------------------------
{
  cudaFree (_cuda_module1);
  cudaFree (_cuda_module2);
  cudaFree (_cuda_z0_min);
  cudaFree (_cuda_z0_max);
  cudaFree (_cuda_dphi_min);
  cudaFree (_cuda_dphi_max);
  cudaFree (_cuda_phi_slope_min);
  cudaFree (_cuda_phi_slope_max);
  cudaFree (_cuda_deta_min);
  cudaFree (_cuda_deta_max);
}

//-----------------------------------------------------------------------------------------------
template <class T>
void CUDA_module_map_doublet<T>::build_doublets (module_map_triplet<T>& MMT, std::string modules)
//-----------------------------------------------------------------------------------------------
{
  _nb_doublets = MMT.size();
  _module_map = MMT.module_map();
  module_doublet<T> *it2;

  for (std::_Rb_tree_iterator<std::pair<const std::vector<uint64_t>, module_triplet<T>>> it3 = MMT.map_triplet().begin(); it3 != MMT.map_triplet().end(); it3++) {
    if (modules == "12") {
      _module1.push_back (_module_map.find (it3->first[0]) -> second);
      _module2.push_back (_module_map.find (it3->first[1]) -> second);
      it2 = &it3->second.modules12();
    }

    if (modules == "23") {
      _module1.push_back (_module_map.find (it3->first[1]) -> second);
      _module2.push_back (_module_map.find (it3->first[2]) -> second);
      it2 = &it3->second.modules23();
    }

    _z0_min.push_back (it2->z0_min());
    _z0_max.push_back (it2->z0_max());
    _dphi_min.push_back (it2->dphi_min());
    _dphi_max.push_back (it2->dphi_max());
    _phi_slope_min.push_back (it2->phi_slope_min());
    _phi_slope_max.push_back (it2->phi_slope_max());
    _deta_min.push_back (it2->deta_min());
    _deta_max.push_back (it2->deta_max());
  }
}

//---------------------------------------------
template <class T>
void CUDA_module_map_doublet<T>::HostToDevice ()
//----------------------------------------------
{
  allocate_doublets();
  // send module map doublets data to GPU
  std::cout << "sending data to GPU" << std::endl;
  cudaMemcpy (_cuda_module1, _module1.data(), _nb_doublets * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy (_cuda_module2, _module2.data(), _nb_doublets * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy (_cuda_z0_min, _z0_min.data(), _nb_doublets * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy (_cuda_z0_max, _z0_max.data(), _nb_doublets * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy (_cuda_dphi_min, _dphi_min.data(), _nb_doublets * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy (_cuda_dphi_max, _dphi_max.data(), _nb_doublets * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy (_cuda_phi_slope_min, _phi_slope_min.data(), _nb_doublets * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy (_cuda_phi_slope_max, _phi_slope_max.data(), _nb_doublets * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy (_cuda_deta_min, _deta_min.data(), _nb_doublets * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy (_cuda_deta_max, _deta_max.data(), _nb_doublets * sizeof(T), cudaMemcpyHostToDevice);
std::cout << "done" << std::endl;
}
