/***************************************************************
 * Tracking project library - L2IT
 * Trace reconstruction in LHC
 * copyright © 2022,2023 ROUGIER Charline
 * copyright © 2022,2023,2024 COLLARD Christophe
 * copyright © 2022,2023,2024 Centre National de la Recherche Scientifique
 * copyright © 2022,2023,2024 Laboratoire des 2 Infinis de Toulouse (L2IT)
 ***************************************************************/

#include "module_doublet.hpp"

//----------------------------------
template <class T>
module_doublet<T>::module_doublet ()
//----------------------------------
{
  _cuts_min = geometric_cuts<T> (-999, -999, -999, -999);
  _cuts_max = geometric_cuts<T> (999, 999, 999, 999);
}


//---------------------------------------------------------------
template <class T>
module_doublet<T>::module_doublet (const geometric_cuts<T>& cuts)
//---------------------------------------------------------------
{
  _cuts_min = _cuts_max = cuts;
}


//-----------------------------------------------------------------------------
template <class Tf>
bool operator == (const module_doublet<Tf>& mt1, const module_doublet<Tf>& mt2)
//-----------------------------------------------------------------------------
{
  return (mt1._cuts_min == mt2._cuts_min) && (mt1._cuts_max == mt2._cuts_max);
}


//-----------------------------------------------------------------------------
template <class Tf>
bool operator != (const module_doublet<Tf>& md1, const module_doublet<Tf>& md2)
//-----------------------------------------------------------------------------
{
  return !(md1 == md2);
}


//-----------------------------------------------------------------------
template <class Tf>
std::ostream& operator << (std::ostream& s, const module_doublet<Tf>& md)
//-----------------------------------------------------------------------
{
//  s << "occurence # " << mt.occurence << std::endl;
  s << "z0 max 12 = " << md._cuts_max.z0() << std::endl;
  s << "z0 min 12 = " << md._cuts_min.z0() << std::endl;
  s << "d_phi max 12 = " << md._cuts_max.d_phi() << std::endl;
  s << "d_phi min 12 = " << md._cuts_min.d_phi() << std::endl;
  s << "phi slope max 12 = " << md._cuts_max.phi_slope() << std::endl;
  s << "phi slope min 12 = " << md._cuts_min.phi_slope() << std::endl;
  s << "delta max 12 = " << md._cuts_max.d_eta() << std::endl;
  s << "delta min 12 = " << md._cuts_min.d_eta() << std::endl;

  return s;
}


//------------------------------------------------------------------------------------
template <class T>
void module_doublet<T>::branch (TTree* treeModuleDoublets, const std::string& modules)
//------------------------------------------------------------------------------------
{
  _cuts_min.branch (treeModuleDoublets, modules, "min");
  _cuts_max.branch (treeModuleDoublets, modules, "max");
}


//------------------------------------------------------------------------------------------------
template <class T>
void module_doublet<T>::set_branch_address (TTree* treeModuleDoublets, const std::string& modules)
//------------------------------------------------------------------------------------------------
{
  _cuts_min.set_branch_address (treeModuleDoublets, modules, "min");
  _cuts_max.set_branch_address (treeModuleDoublets, modules, "max");
}


//-------------------------------------------------------------------
template <class T>
bool module_doublet<T>::add_occurence (const geometric_cuts<T>& cuts)
//-------------------------------------------------------------------
{
  if (!cuts.check_cut_values()) return false;
  _cuts_min.min (cuts);
  _cuts_max.max (cuts);

  return true;
}


//----------------------------------------------------------------------------
template <class T>
void module_doublet<T>::add_occurence (const module_doublet<T>& ModuleDoublet)
//----------------------------------------------------------------------------
{
  _cuts_min.min (ModuleDoublet._cuts_min);
  _cuts_max.max (ModuleDoublet._cuts_max);
}
