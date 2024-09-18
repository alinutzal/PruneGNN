/***************************************************************
 * Tracking project library - L2IT
 * Trace reconstruction in LHC
 * copyright © 2023 ROUGIER Charline
 * copyright © 2023,2024 COLLARD Christophe
 * copyright © 2023,2024 Centre National de la Recherche Scientifique
 * copyright © 2023,2024 Laboratoire des 2 Infinis de Toulouse (L2IT)
 ***************************************************************/

#include "geometric_cuts.hpp"


//---------------------------------------------------------------------
template<class T>
geometric_cuts<T>::geometric_cuts (T z0_, T dphi, T phi_slope_, T deta)
//---------------------------------------------------------------------
{
  _z0 = z0_;
  _dphi = dphi;
  _phi_slope = phi_slope_;
  _deta = deta;
}

//---------------------------------------------------------------------------------------------
template<class T>
template<class V>
geometric_cuts<T>::geometric_cuts (const TTree_hits<V>& TThits, const int& it1, const int& it2)
//---------------------------------------------------------------------------------------------
{
  T R1 = TThits.R(it1);
  T z1 = TThits.z(it1);
//  char hardware1 = (TThits.hardware() == "PIXEL") ? 1 : 2;

  T R2 = TThits.R(it2);
  T z2 = TThits.z(it2);
//  char hardware2 = (TThits.hardware() == "PIXEL") ? 1 : 2;

  T dr   = R2 - R1;
  T R = 0.5 * (R1 + R2);

//  strip_hit_pair pair ();
//  pair.reconstruction();

//  if (R1 > R2) -> R2 = in
//    pair = new StripHitPair (StripHitPair::Hit (in_hardware, in_barrel_endcap, in_layer_disk,
//				      in_spoint, in_cluster1, in_cluster2, in_module),
//		    StripHitPair::Hit(out_hardware, out_barrel_endcap, out_layer_disk,
//				      out_spoint, out_cluster1, out_cluster2, out_module));

  T dz = z2 - z1;
  _deta = TThits.Eta(it1) - TThits.Eta(it2);
  _dphi = TThits.Phi(it2) - TThits.Phi(it1);

  if (_dphi > TMath::Pi()) _dphi -= 2 * TMath::Pi();
  else if (_dphi < -TMath::Pi()) _dphi += 2 * TMath::Pi();

  if (dr) {
    _phi_slope = _dphi / dr;
    _r_phi_slope = R * _phi_slope;
    _z0 = z1 - R1 * dz / dr;
  }
  else {
    _z0 = (dz>=0) ? std::numeric_limits<T>::max() : -std::numeric_limits<T>::max();
    _phi_slope = _dphi * std::numeric_limits<T>::max();
    _r_phi_slope = _dphi * std::numeric_limits<T>::max();
//      _z0 = TMath::Sign(std::numeric_limits<T>::max(), dz);
      // if dr==0 but dphi non null : put float highest value 
      // if dr==0 && dphi == 0 : put 0 (phi do not change) 
//      if (_dphi > 0)
//        _phi_slope = std::numeric_limits<T>::max();
//      else if (_dphi < 0)
//        _phi_slope = -std::numeric_limits<T>::max();
//      else
//        _phi_slope = 0;
    }
}


//------------------------------------------------------------------------------------------------------------
template <class T>
void geometric_cuts<T>::branch (TTree* treeGeometricCuts, const std::string& modules, const std::string& name)
//------------------------------------------------------------------------------------------------------------
{
  treeGeometricCuts -> Branch ((std::string("z0")+name+std::string("_")+modules).c_str(), &_z0);
  treeGeometricCuts -> Branch ((std::string("dphi")+name+std::string("_")+modules).c_str(), &_dphi);
  treeGeometricCuts -> Branch ((std::string("phiSlope")+name+std::string("_")+modules).c_str(), &_phi_slope);
  treeGeometricCuts -> Branch ((std::string("deta")+name+std::string("_")+modules).c_str(), &_deta);
}


//------------------------------------------------------------------------------------------------------------------------
template <class T>
void geometric_cuts<T>::set_branch_address (TTree* treeGeometricCuts, const std::string& modules, const std::string& name)
//------------------------------------------------------------------------------------------------------------------------
{
  treeGeometricCuts -> SetBranchAddress ((std::string("z0")+name+std::string("_")+modules).c_str(), &_z0);
  treeGeometricCuts -> SetBranchAddress ((std::string("dphi")+name+std::string("_")+modules).c_str(), &_dphi);
  treeGeometricCuts -> SetBranchAddress ((std::string("phiSlope")+name+std::string("_")+modules).c_str(), &_phi_slope);
  treeGeometricCuts -> SetBranchAddress ((std::string("deta")+name+std::string("_")+modules).c_str(), &_deta);
}


//-----------------------------------------------------------------------------
template <class Tf>
bool operator == (const geometric_cuts<Tf>& ct1, const geometric_cuts<Tf>& ct2)
//-----------------------------------------------------------------------------
{
  return (ct1._z0 == ct2._z0) && (ct1._dphi == ct2._dphi) && (ct1._phi_slope == ct2._phi_slope) && (ct1._deta == ct2._deta);
}


//-----------------------------------------------------------------------------
template <class Tf>
bool operator != (const geometric_cuts<Tf>& ct1, const geometric_cuts<Tf>& ct2)
//-----------------------------------------------------------------------------
{
  return !(ct1==ct2);
}


//-----------------------------------------------------------------------
template <class Tf>
std::ostream& operator << (std::ostream& s, const geometric_cuts<Tf>& ct)
//-----------------------------------------------------------------------
{
  s << "z0 = " << ct._z0 << std::endl;
  s << "Dphi = " << ct._dphi << std::endl;
  s << "Phi Slope = " << ct._phi_slope << std::endl;
  s << "Deta = " << ct._deta << std::endl;

  return s;
}


//-----------------------------------------------
template<class T>
bool geometric_cuts<T>::check_cut_values () const
//-----------------------------------------------
{
  if (isnan (_z0)) {
    std::cout << "Error: z0 Not A Number, you cannot add the link to the module map." << std::endl;
    return false;
  }

  if (isnan (_dphi)) {
    std::cout << "Error: dphi Not A Number, you cannot add the link to the module map." << std::endl;
    return false;
  }

  if (isnan (_phi_slope)) {
    std::cout << "Error: phi_slope Not A Number, you cannot add the link to the module map." << std::endl;
    return false;
  }

  if (isnan(_deta)) {
    std::cout << "Error: deta Not A Number, you cannot add the link to the module map." << std::endl;
    return false;
  }

  if (isinf (_z0)) {
    std::cout << "Error: z0 is infinite, you cannot add the link to the module map. " << std::endl;
    return false;
  }

  if (isinf (_dphi)) {
    std::cout << "Error: dphi is infinite, you cannot add the link to the module map." << std::endl;
    return false;
  }

  if (isinf (_phi_slope)) {
    std::cout << "Error: phi_slope is infinite, you cannot add the link to the module map." << std::endl;
    return false;
  }

  if (isinf(_deta)) {
    std::cout << "Error: deta is infinite, you cannot add the link to the module map." << std::endl;
    return false;
  }

  return true;
}


//---------------------------------------------------------
template<class T>
void geometric_cuts<T>::min (const geometric_cuts<T>& cuts)
//---------------------------------------------------------
{
  if (_z0 > cuts.z0()) _z0 = cuts.z0();
  if (_dphi > cuts.d_phi()) _dphi = cuts.d_phi();
  if (_phi_slope > cuts.phi_slope()) _phi_slope = cuts.phi_slope();
  if (_deta > cuts.d_eta()) _deta = cuts.d_eta();
}


//---------------------------------------------------------
template<class T>
void geometric_cuts<T>::max (const geometric_cuts<T>& cuts)
//---------------------------------------------------------
{
  if (_z0 < cuts.z0()) _z0 = cuts.z0();
  if (_dphi < cuts.d_phi()) _dphi = cuts.d_phi();
  if (_phi_slope < cuts.phi_slope()) _phi_slope = cuts.phi_slope();
  if (_deta < cuts.d_eta()) _deta = cuts.d_eta();
}
