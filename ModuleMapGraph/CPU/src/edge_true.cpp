/***************************************************************
 * Tracking project library - L2IT
 * Trace reconstruction in LHC
 * copyright © 2022,2023 ROUGIER Charline
 * copyright © 2022,2023 COLLARD Christophe
 * copyright © 2022,2023 Centre National de la Recherche Scientifique
 * copyright © 2022,2023 Laboratoire des 2 Infinis de Toulouse (L2IT)
 ***************************************************************/

#include "edge_true.hpp"

//-----------------------------------------------------------------------------------------------------------------
template <class T>
edge_true<T>::edge_true (const int& is_segment_true, const T& pt_particle, const int& mask_edge, const int& region)
//-----------------------------------------------------------------------------------------------------------------
{
  _is_segment_true = is_segment_true;
  _pt_particle = pt_particle;
  _mask_edge = mask_edge;
  _region = region;
}


//------------------------------------------------------------
template <class T>
edge_true<T>& edge_true<T>::operator = (const edge_true<T>& e)
//------------------------------------------------------------
{
  _is_segment_true = e._is_segment;
  _pt_particle = e._pt_particle;
  _mask_edge = e._mask_edge;
  _region = e._region;

  return (*this);
}
