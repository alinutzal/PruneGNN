/***************************************************************
 * Tracking project library - L2IT
 * Trace reconstruction in LHC
 * copyright © 2024 COLLARD Christophe
 * copyright © 2024 Centre National de la Recherche Scientifique
 * copyright © 2024 Laboratoire des 2 Infinis de Toulouse (L2IT)
 ***************************************************************/

#include "TTree_hits.cuh"

//=====GPU functions for CUDA TTree hits=====================================

//-------------------------------------------------------------------------------------------------------
template <class T>
__global__ void TTree_hits_constants (int nb_hits, T* x, T* y, T* z, T* cuda_R, T* cuda_eta, T* cuda_phi)
//-------------------------------------------------------------------------------------------------------
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb_hits) return;

  data_type2 xi = x[i];
  data_type2 yi = y[i];
  data_type2 zi = z[i];
//  data_type2 R = std::sqrt (pow (xi, 2) + pow (yi, 2));
  data_type2 R2 = pow (xi, 2) + pow (yi, 2);
  cuda_R[i] = std::sqrt (R2);
//  data_type2 r3_inv = std::sqrt (1 + pow (zi / (xi+yi), 2));

//  data_type2 r3 = std::sqrt (pow(R, 2) + pow(zi,2));
  data_type2 r3 = std::sqrt (R2 + zi*zi);
  data_type2 theta = (data_type2) 0.5 * acos (zi / r3);
  data_type2 eta = -log (tan (theta));
  cuda_eta[i] = eta;

  cuda_phi[i] = atan2 (yi, xi);
}

//=====Private methods for TTree hits========================================

//=====Public methods for TTree hits=========================================


//--------------------------------------------------------------------------------------------------------
template <class T>
CUDA_TTree_hits<T>::CUDA_TTree_hits (TTree_hits<T>& TThits, const std::multimap<uint64_t,int>& module_map)
//--------------------------------------------------------------------------------------------------------
{
  _size = TThits.size();
  _position = 0;

  cudaMalloc (&_cuda_hit_id, _size * sizeof(uint64_t));
  cudaMalloc (&_cuda_x, _size * sizeof(T));
  cudaMalloc (&_cuda_y, _size * sizeof(T));
  cudaMalloc (&_cuda_z, _size * sizeof(T));
  cudaMalloc (&_cuda_R, _size * sizeof(T));
  cudaMalloc (&_cuda_eta, _size * sizeof(T));
  cudaMalloc (&_cuda_phi, _size * sizeof(T));
  cudaMalloc (&_cuda_vertices, _size * sizeof(int));
  _R = _eta = _phi = std::vector<T> (_size);

  std::vector<int> nb_hits (module_map.size(), 0);
  for (std::pair<uint64_t, int> map : TThits.hits_map()) {
    std::_Rb_tree_const_iterator<std::pair<const uint64_t, int>> it = module_map.find (map.first);
    if (it != module_map.end()) {
      int hit_location = map.second;
      if (std::find (_hit_id.begin(), _hit_id.end(), TThits.hit_id(hit_location)) == _hit_id.end()) {
        nb_hits[it->second] += 1;
        _hit_id.push_back (TThits.hit_id (hit_location));
        _x.push_back (TThits.x (hit_location));
        _y.push_back (TThits.y (hit_location));
        _z.push_back (TThits.z (hit_location));
      }
    }
  }

  _hit_indice.push_back (0);
  for (int i=0; i<nb_hits.size(); i++)
    _hit_indice.push_back (_hit_indice[i] + nb_hits[i]);

  _vertices = std::vector<int> (_size, false);
  cudaMalloc (&_cuda_hit_indice, _hit_indice.size() * sizeof(int));
}


//-------------------------------------
template <class T>
CUDA_TTree_hits<T>::~CUDA_TTree_hits ()
//-------------------------------------
{
  cudaFree (_cuda_hit_indice);
  cudaFree (_cuda_vertices);
  cudaFree (_cuda_phi);
  cudaFree (_cuda_eta);
  cudaFree (_cuda_R);
  cudaFree (_cuda_z);
  cudaFree (_cuda_y);
  cudaFree (_cuda_x);
  cudaFree (_cuda_hit_id);
}


//--------------------------------------
template <class T>
void CUDA_TTree_hits<T>::DeviceToHost ()
//--------------------------------------
{
  cudaMemcpy (_R.data(), _cuda_R, _size * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy (_eta.data(), _cuda_eta, _size * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy (_phi.data(), _cuda_phi, _size * sizeof(T), cudaMemcpyDeviceToHost);
}


//--------------------------------------
template <class T>
void CUDA_TTree_hits<T>::HostToDevice ()
//--------------------------------------
{
  assert (_vertices.size() && _size>0);
  cudaMemcpy (_cuda_hit_id, _hit_id.data(), _size * sizeof(uint64_t), cudaMemcpyHostToDevice);
  cudaMemcpy (_cuda_x, _x.data(), _size * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy (_cuda_y, _y.data(), _size * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy (_cuda_z, _z.data(), _size * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy (_cuda_hit_indice, _hit_indice.data(), _hit_indice.size() * sizeof (int), cudaMemcpyHostToDevice);
  cudaMemcpy (_cuda_vertices, _vertices.data(), _size * sizeof (int), cudaMemcpyHostToDevice);
}
