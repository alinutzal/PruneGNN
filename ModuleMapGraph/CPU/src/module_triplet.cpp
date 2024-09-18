/***************************************************************
 * Tracking project library - L2IT
 * Trace reconstruction in LHC
 * copyright © 2022,2023 ROUGIER Charline
 * copyright © 2022,2023,2024 COLLARD Christophe
 * copyright © 2022,2023,2024 Centre National de la Recherche Scientifique
 * copyright © 2022,2023,2024 Laboratoire des 2 Infinis de Toulouse (L2IT)
 ***************************************************************/

#include "module_triplet.hpp"

//----------------------------------
template <class T>
module_triplet<T>::module_triplet ()
//----------------------------------
{
  _occurence = 0;
  _diff_dzdr_min = 999;
  _diff_dzdr_max = -999;
  _diff_dydx_min = 999;
  _diff_dydx_max = -999;
}


//------------------------------------------------------------------------------------------------------------------------------------------
template <class T>
module_triplet<T>::module_triplet (const geometric_cuts<T>& cuts12, const geometric_cuts<T>& cuts23, const T& diff_dzdr, const T& diff_dydx)
//------------------------------------------------------------------------------------------------------------------------------------------
{
  geometric_cuts<T> cuts12d (cuts12.z0(), cuts12.d_phi(), cuts12.phi_slope(), cuts12.d_eta());
  geometric_cuts<T> cuts23d (cuts23.z0(), cuts23.d_phi(), cuts23.phi_slope(), cuts23.d_eta());
  _MD12 = module_doublet<T> (cuts12d);
  _MD23 = module_doublet<T> (cuts23d);
//  _MD12 = module_doublet<double> (cuts12);

//  _MD23 = module_doublet<double> (cuts23);
  _diff_dzdr_min = _diff_dzdr_max = diff_dzdr;
  _diff_dydx_min = _diff_dydx_max = diff_dydx;
  _occurence = 1;
}


//-----------------------------------------------------------------------------
template <class Tf>
bool operator == (const module_triplet<Tf>& mt1, const module_triplet<Tf>& mt2)
//-----------------------------------------------------------------------------
{
  bool test = (mt1._occurence == mt2._occurence);
  test *= (mt1._MD12 == mt2._MD12);
  test *= (mt1._MD23 == mt2._MD23);
  test *= (mt1._diff_dzdr_max == mt2._diff_dzdr_max)  &&  (mt1._diff_dzdr_min == mt2._diff_dzdr_min)  &&  (mt1._diff_dydx_max == mt2._diff_dydx_max)  &&  (mt1._diff_dydx_min == mt2._diff_dydx_min);

  return test;
}


//-----------------------------------------------------------------------------
template <class Tf>
bool operator != (const module_triplet<Tf>& mt1, const module_triplet<Tf>& mt2)
//-----------------------------------------------------------------------------
{
  return !(mt1 == mt2);
}


//-----------------------------------------------------------------------
template <class Tf>
std::ostream& operator << (std::ostream& s, const module_triplet<Tf>& mt)
//-----------------------------------------------------------------------
{
  s << "occurence # " << mt._occurence << std::endl;
  s << mt._MD12;
  s << mt._MD23;
  s << "diff dy dx max = " << mt._diff_dydx_min << std::endl;
  s << "diff dy dx max = " << mt._diff_dydx_max << std::endl;
  s << "diff dz dr min = " << mt._diff_dzdr_min << std::endl;
  s << "diff dz dr max = " << mt._diff_dzdr_max << std::endl;

  return s;
}


//--------------------------------------------------------
template <class T>
void module_triplet<T>::branch (TTree* treeModuleTriplets)
//--------------------------------------------------------
{
  treeModuleTriplets -> Branch ("Occurence", &_occurence, "Occurence/i");
  _MD12.branch (treeModuleTriplets, "12");
  _MD23.branch (treeModuleTriplets, "23");
  treeModuleTriplets -> Branch ("diff_dzdr_max", &_diff_dzdr_max);
  treeModuleTriplets -> Branch ("diff_dzdr_min", &_diff_dzdr_min);
  treeModuleTriplets -> Branch ("diff_dydx_max", &_diff_dydx_max);
  treeModuleTriplets -> Branch ("diff_dydx_min", &_diff_dydx_min);
}


//--------------------------------------------------------------------
template <class T>
void module_triplet<T>::set_branch_address (TTree* treeModuleTriplets)
//--------------------------------------------------------------------
{
  treeModuleTriplets -> SetBranchAddress ("Occurence", &_occurence);
  _MD12.set_branch_address (treeModuleTriplets, "12");
  _MD23.set_branch_address (treeModuleTriplets, "23");
  treeModuleTriplets -> SetBranchAddress ("diff_dzdr_max", &_diff_dzdr_max);
  treeModuleTriplets -> SetBranchAddress ("diff_dzdr_min", &_diff_dzdr_min);
  treeModuleTriplets -> SetBranchAddress ("diff_dydx_max", &_diff_dydx_max);
  treeModuleTriplets -> SetBranchAddress ("diff_dydx_min", &_diff_dydx_min);
}


//--------------------------------------
template <class T>
bool module_triplet<T>::min_equal_max ()
//--------------------------------------
{
  return (_diff_dydx_min == _diff_dydx_max) && (_diff_dzdr_min == _diff_dzdr_max) && _MD12.min_equal_max() && _MD23.min_equal_max();
}


//----------------------------------------------------------------------------------------------------------------------------------------------
template <class T>
bool module_triplet<T>::add_occurence (const geometric_cuts<T>& cuts12, const geometric_cuts<T>& cuts23, const T& diff_dzdr, const T& diff_dydx)
//----------------------------------------------------------------------------------------------------------------------------------------------
{
  if (isnan (diff_dydx)) {
    std::cout << "Error : diff_dydx Not A Number , you cannot add the link to the module map." << std::endl;
    return false;
  }

  if (isnan (diff_dzdr)) {
    std::cout << "Error : diff_dzdr Not A Number , you cannot add the link to the module map." << std::endl;
    return false;
  }

  if (isinf (diff_dzdr)) {
    std::cout << "Error : diff_dzdr is infinite , you cannot add the link to the module map." << std::endl;
    return false;
  }

  if (isinf(diff_dydx)) {
    std::cout << "Error : diff_dydx is infinite , you cannot add the link to the module map." << std::endl;
    return false;
  }

  if (_diff_dzdr_max < diff_dzdr) _diff_dzdr_max = diff_dzdr;
  if (_diff_dzdr_min > diff_dzdr) _diff_dzdr_min = diff_dzdr;

  if (_diff_dydx_max < diff_dydx) _diff_dydx_max = diff_dydx;
  if (_diff_dydx_min > diff_dydx) _diff_dydx_min = diff_dydx;

  _MD12.add_occurence (cuts12);
  _MD23.add_occurence (cuts23);

  _occurence++;

  return true;
}


//----------------------------------------------------------------------------
template <class T>
void module_triplet<T>::add_occurence (const module_triplet<T>& ModuleTriplet)
//----------------------------------------------------------------------------
{
  if (_diff_dzdr_max < ModuleTriplet._diff_dzdr_max) _diff_dzdr_max = ModuleTriplet._diff_dzdr_max;
  if (_diff_dzdr_min > ModuleTriplet._diff_dzdr_min) _diff_dzdr_min = ModuleTriplet._diff_dzdr_min;

  if (_diff_dydx_max < ModuleTriplet._diff_dydx_max) _diff_dydx_max = ModuleTriplet._diff_dydx_max;
  if (_diff_dydx_min > ModuleTriplet._diff_dydx_min) _diff_dydx_min = ModuleTriplet._diff_dydx_min;

  _MD12.add_occurence (ModuleTriplet.modules12());
  _MD23.add_occurence (ModuleTriplet.modules23());

  _occurence += ModuleTriplet._occurence;
}
