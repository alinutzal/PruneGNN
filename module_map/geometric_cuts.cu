/***************************************************************
 * Tracking project library - L2IT
 * Trace reconstruction in LHC
 * copyright © 2024 COLLARD Christophe
 * copyright © 2024 Centre National de la Recherche Scientifique
 * copyright © 2024 Laboratoire des 2 Infinis de Toulouse (L2IT)
 ***************************************************************/

//----------------------------------------------------------------------------------------------------------------------------------------------------------------
template <class T>
__global__ void hits_geometric_cuts (T *z0, T *phi_slope, T *deta, T *dphi, int *SPi, int *SPo, T *R, T *z, T *eta, T *phi, data_type2 pi, T max, int nb_doublets)
//----------------------------------------------------------------------------------------------------------------------------------------------------------------
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb_doublets) return;

  int SP1 = SPi[i];
  int SP2 = SPo[i];
  T R1 = R[SP1];
  T R2 = R[SP2];
  T z1 = z[SP1];
  T z2 = z[SP2];

  T dr = R2 - R1;
  T dz = z2 - z1;
  deta[i] = eta[SP1] - eta[SP2];
  T dphi_loc = phi[SP2] - phi[SP1];
  T pipi = (data_type2) 2 * pi;

  if (dphi_loc > pi) dphi_loc -= pipi;
  else if (dphi_loc < -pi) dphi_loc += pipi;

  if (dr) {
    phi_slope[i] = dphi_loc / dr;
    z0[i] = z1 - R1 * dz / dr;
  }
  else {
    z0[i] = (dz < 0) ? -max : max;
    phi_slope[i] = (dphi_loc < 0) ? -max : max;
  }

  dphi[i] = dphi_loc;
}

//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <class T>
__device__ void hits_geometric_cuts (const T& R1, const T& R2, const T& z1, const T z2, const T& eta1, const T& eta2, const T& phi1, const T& phi2, const data_type2& pi, const T& max, T& z0, T& phi_slope, T& deta, T& dphi)
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
{
  T dr = R2 - R1;
  T dz = z2 - z1;
  deta = eta1 - eta2;
  dphi = phi2 - phi1;

  if (dphi > pi) dphi -= (data_type2) 2 * pi;
  else if (dphi < -pi) dphi += (data_type2) 2 * pi;

  if (dr) {
    phi_slope = dphi / dr;
    z0 = z1 - R1 * dz / dr;
  }
  else {
    z0 = (dz < 0) ? -max : max;
    phi_slope = (dphi < 0) ? -max : max;
  }
}

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <class T>
__device__ bool apply_geometric_cuts (int i, const T& z0, const T& phi_slope, const T& deta, const T& dphi, T *z0_min, T *z0_max, T *deta_min, T *deta_max, T *phi_slope_min, T *phi_slope_max, T *dphi_min, T *dphi_max)
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
{
  bool accept = (z0_min[i] <= z0) * (dphi_min[i] <= dphi) * (phi_slope_min[i] <= phi_slope) * (deta_min[i] <= deta);
  accept *= (z0 <= z0_max[i]) * (dphi <= dphi_max[i]) * (phi_slope <= phi_slope_max[i]) * (deta <= deta_max[i]);

  return accept;
}

//----------------------------------------------------------------------------------------------
template <class T>
__device__ T Diff_dydx (T *x, T *y, T *z, const int& it1, const int& it2, const int& it3, T max)
//----------------------------------------------------------------------------------------------
{
  T dy_12 = y[it2] - y[it1];
  T dy_23 = y[it2] - y[it3];
  T dx_12 = x[it1] - x[it2];
  T dx_23 = x[it2] - x[it3];

  T diff_dydx = 0;

  if (dx_12 && dx_23)
    diff_dydx = (dy_12 / dx_12) - (dy_23 / dx_23);
  else if (dx_12)
    diff_dydx = (dy_12 * dx_12 > 0) ? max : -max;
  else if (dx_23)
    diff_dydx = (dy_23 * dx_23 < 0) ? max : -max;

    return diff_dydx;
}

//----------------------------------------------------------------------------------------
template <class T>
__device__ T Diff_dzdr (T *R, T *z, const int& it1, const int& it2, const int& it3, T max)
//----------------------------------------------------------------------------------------
{
  T dz_12 = z[it2] - z[it1];
  T dz_23 = z[it3] - z[it2];
  T dr_12 = R[it2] - R[it1];
  T dr_23 = R[it3] - R[it2];

  T diff_dzdr = 0;

  if (dr_12 && dr_23)
    diff_dzdr = (dz_12 / dr_12) - (dz_23 / dr_23);
  else if (dr_12)
    diff_dzdr = (dz_12 * dr_12 >= 0) ? max : -max;
  else if (dr_23)
    diff_dzdr = (dz_23 * dr_23 < 0) ? max : -max;

    return diff_dzdr;
}

//------------------------------------------------------------------------------------------------------------
template <class T>
__global__ void doublet_cuts (int nb_doublets, int *modules1, int *modules2, T *R, T *z, T *eta, T *phi,
                              T *z0_min, T *z0_max, T *deta_min, T *deta_max,
                              T *phi_slope_min, T *phi_slope_max, T *dphi_min, T *dphi_max,
                              int *indices, T pi, T max, int *M1_SP, int *M2_SP, int *nb_edges, int max_edges)
//------------------------------------------------------------------------------------------------------------
{
  // loop over module1 SP
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb_doublets) return;

  int module1 = modules1[i];
  int module2 = modules2[i];
  int shift = i * max_edges;
  int edges = shift;

  for (int k = indices[module1]; k < indices[module1+1]; k++) {
    T phi_SP1 = phi[k];
    T eta_SP1 = eta[k];
    T R_SP1 = R[k];
    T z_SP1 = z[k];

    for (int l = indices[module2]; l < indices[module2+1]; l++) {
      T z0, phi_slope, deta, dphi;
      hits_geometric_cuts<T> (R_SP1, R[l], z_SP1, z[l], eta_SP1, eta[l], phi_SP1, phi[l], pi, max, z0, phi_slope, deta, dphi);

      if (apply_geometric_cuts (i, z0, phi_slope, deta, dphi, z0_min, z0_max, deta_min, deta_max, phi_slope_min, phi_slope_max, dphi_min, dphi_max)) {
        M1_SP[edges] = k;
        M2_SP[edges] = l;
        edges++;
      }
    }
  }

  nb_edges[i] = edges-shift;
}

//---------------------------------------------------------------------------------------------------------------------------------------------------
template <class T>
__global__ void triplet_cuts (int nb_triplets, int *modules12_map, int *modules23_map, T *x, T *y, T *z, T *R,
                              T *z0, T *phi_slope, T *deta, T *dphi,
                              T *MD12_z0_min, T *MD12_z0_max, T *MD12_deta_min, T *MD12_deta_max,
                              T *MD12_phi_slope_min, T *MD12_phi_slope_max, T *MD12_dphi_min, T *MD12_dphi_max,
                              T *MD23_z0_min, T *MD23_z0_max, T *MD23_deta_min, T *MD23_deta_max,
                              T *MD23_phi_slope_min, T *MD23_phi_slope_max, T *MD23_dphi_min, T *MD23_dphi_max,
                              T *diff_dydx_min, T *diff_dydx_max, T *diff_dzdr_min, T *diff_dzdr_max,
                              T pi, T max, int *M1_SP, int *M2_SP, int* sorted_M2_SP, int* edge_indices, int *vertices, int *edge_tag)
//---------------------------------------------------------------------------------------------------------------------------------------------------
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb_triplets) return;

  int module12 = modules12_map[i];
  int module23 = modules23_map[i];

  int nb_hits_M12 = edge_indices[module12 + 1] - edge_indices[module12];
  int nb_hits_M23 = edge_indices[module23 + 1] - edge_indices[module23];

  bool hits_on_modules = nb_hits_M12 * nb_hits_M23;
  if (!hits_on_modules) return;

  int shift12 = edge_indices[module12];
  int shift23 = edge_indices[module23];

  int last12 = shift12 + nb_hits_M12 - 1;
  int ind23 = shift23 + nb_hits_M23;
  for (int k=shift12; k<=last12; k++) {
    int p = sorted_M2_SP[k];
    int SP1 = M1_SP[p];
    int SP2 = M2_SP[p];
    bool next_ind = false;
    if (k < last12) next_ind = (SP2 != (M2_SP[sorted_M2_SP[k+1]]));

    if (!apply_geometric_cuts (i, z0[p], phi_slope[p], deta[p], dphi[p], MD12_z0_min, MD12_z0_max, MD12_deta_min, MD12_deta_max, MD12_phi_slope_min, MD12_phi_slope_max, MD12_dphi_min, MD12_dphi_max)) continue;

    int l = shift23;
    for (; l<ind23 && SP2 != M1_SP[l]; l++); // search first hit indice on M23_1 = M12_2

    bool new_elt = false;
    for (; l<ind23 && SP2 == M1_SP[l]; l++) {
      int SP3 = M2_SP[l];
      if (!apply_geometric_cuts (i, z0[l], phi_slope[l], deta[l], dphi[l], MD23_z0_min, MD23_z0_max, MD23_deta_min, MD23_deta_max, MD23_phi_slope_min, MD23_phi_slope_max, MD23_dphi_min, MD23_dphi_max)) continue;

      T diff_dydx = Diff_dydx (x, y, z, SP1, SP2, SP3, max);
      if (!((diff_dydx >= diff_dydx_min[i]) * (diff_dydx <= diff_dydx_max[i]))) continue;

      T diff_dzdr = Diff_dzdr (R, z, SP1, SP2, SP3, max);
      if (! ((diff_dzdr >= diff_dzdr_min[i]) * (diff_dzdr <= diff_dzdr_max[i]))) continue;

      vertices[SP3] = edge_tag[l] = true;
      new_elt = true;
    }
    if (new_elt) edge_tag[p] = vertices[SP1] = vertices[SP2] = true;
    if (next_ind && new_elt) shift23 = l;
  }
}
