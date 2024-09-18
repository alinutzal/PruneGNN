/***************************************************************
 * Tracking project library - L2IT
 * Trace reconstruction in LHC
 * copyright © 2022,2023 ROUGIER Charline
 * copyright © 2022,2023,2024 COLLARD Christophe
 * copyright © 2022,2023,2024 Centre National de la Recherche Scientifique
 * copyright © 2022,2023,2024 Laboratoire des 2 Infinis de Toulouse (L2IT)
 ***************************************************************/

#include "vertex.hpp"

//-------------------------------------------------------------------
template <class T>
vertex<T>::vertex (const T& r, const T& phi, const T& z, const T& id)
//-------------------------------------------------------------------
{
  _r = r;
  _phi = phi;
  _z = z;
  _hit_id = id;
}

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <class T>
vertex<T>::vertex (const T& r, const T& phi, const T& z, const T& eta, const T& r_cluster1, const T& phi_cluster1, const T& z_cluster1, const T& eta_cluster1, const T& r_cluster2, const T& phi_cluster2, const T& z_cluster2, const T& eta_cluster2, const T& id)
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
{
  _r = r;
  _phi = phi;
  _z = z;
  _eta = eta;
  _r_cluster1 = r_cluster1;
  _phi_cluster1 = phi_cluster1;
  _z_cluster1 = z_cluster1;
  _eta_cluster1 = eta_cluster1;
  _r_cluster2 = r_cluster2;
  _phi_cluster2 = phi_cluster2;
  _z_cluster2 = z_cluster2;
  _eta_cluster2 = eta_cluster2;
  _hit_id = id;
}
