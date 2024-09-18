/***************************************************************
 * Tracking project library - L2IT
 * Trace reconstruction in LHC
 * copyright © 2024 COLLARD Christophe
 * copyright © 2024 Centre National de la Recherche Scientifique
 * copyright © 2024 Laboratoire des 2 Infinis de Toulouse (L2IT)
 ***************************************************************/

#include "module_map_triplet.cuh"


//=====Private methods for cuda module map triplet======================================

//---------------------------------------------------
template <class T>
void CUDA_module_map_triplet<T>::allocate_triplets ()
//---------------------------------------------------
{
  cudaMalloc (&_cuda_diff_dydx_min, _nb_triplets * sizeof(T));
  cudaMalloc (&_cuda_diff_dydx_max, _nb_triplets * sizeof(T));
  cudaMalloc (&_cuda_diff_dzdr_min, _nb_triplets * sizeof(T));
  cudaMalloc (&_cuda_diff_dzdr_max, _nb_triplets * sizeof(T));
  cudaMalloc (&_cuda_MD12_map, _nb_triplets * sizeof(int));
  cudaMalloc (&_cuda_MD23_map, _nb_triplets * sizeof(int));
}


//=====Public methods for cuda module map triplet=======================================


//------------------------------------------------------------------------------
template <class T>
CUDA_module_map_triplet<T>::CUDA_module_map_triplet (module_map_triplet<T>& MMT)
//------------------------------------------------------------------------------
{
  _nb_triplets = MMT.map_triplet().size();

  _MD12.build_doublets (MMT, "12");
  _MD23.build_doublets (MMT, "23");

  for (std::_Rb_tree_iterator<std::pair<const std::vector<uint64_t>, module_triplet<T>>> it3 = MMT.map_triplet().begin(); it3 != MMT.map_triplet().end(); it3++) {
    _MD12_map.push_back (MMT.map_pairs().find (std::vector<uint64_t>{it3->first[0], it3->first[1]}) -> second);
    _MD23_map.push_back (MMT.map_pairs().find (std::vector<uint64_t>{it3->first[1], it3->first[2]}) -> second);
    _diff_dydx_min.push_back (it3->second.diff_dydx_min());
    _diff_dydx_max.push_back (it3->second.diff_dydx_max());
    _diff_dzdr_min.push_back (it3->second.diff_dzdr_min());
    _diff_dzdr_max.push_back (it3->second.diff_dzdr_max());
    _occurence.push_back (it3->second.occurence());
  }
}

//-----------------------------------------------------
template <class T>
CUDA_module_map_triplet<T>::~CUDA_module_map_triplet ()
//-----------------------------------------------------
{
  cudaFree (_cuda_diff_dydx_min);
  cudaFree (_cuda_diff_dydx_max);
  cudaFree (_cuda_diff_dzdr_min);
  cudaFree (_cuda_diff_dzdr_max);
  cudaFree (_cuda_MD12_map);
  cudaFree (_cuda_MD23_map);
}

//----------------------------------------------
template <class T>
void CUDA_module_map_triplet<T>::HostToDevice ()
//----------------------------------------------
{
  allocate_triplets();
  // send module map doublets data to GPU
  std::cout << "sending data to GPU" << std::endl;
  _MD12.HostToDevice();
  _MD23.HostToDevice();
  cudaMemcpy (_cuda_MD12_map, _MD12_map.data(), _nb_triplets * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy (_cuda_MD23_map, _MD23_map.data(), _nb_triplets * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy (_cuda_diff_dydx_min, _diff_dydx_min.data(), _nb_triplets * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy (_cuda_diff_dydx_max, _diff_dydx_max.data(), _nb_triplets * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy (_cuda_diff_dzdr_min, _diff_dzdr_min.data(), _nb_triplets * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy (_cuda_diff_dzdr_max, _diff_dzdr_max.data(), _nb_triplets * sizeof(T), cudaMemcpyHostToDevice);
  std::cout << "done" << std::endl;
}
