/***************************************************************
 * Tracking project library - L2IT
 * Trace reconstruction in LHC
 * copyright © 2023 TORRES Heberth
 * copyright © 2023 Centre National de la Recherche Scientifique
 * copyright © 2023 Laboratoire des 2 Infinis de Toulouse (L2IT)
 ***************************************************************/

#include <strip_module_DB.hpp>


//-----------------------------------------------------------------
template <class T>
strip_module_DB<T>::strip_module_DB (const std::string& dbFilePath)
//-----------------------------------------------------------------
{
  TChain tree ("stripModules");
  TString TdbFilePath (dbFilePath);
  tree.Add (TdbFilePath);
  tree.SetEstimate (tree.GetEntries()+1);

  nMods = tree.Draw ("x0:y0:z0:barrel_endcap:layer_disk:eta_module:phi_module", "", "goff");

  for (int i=0; i<nMods; i++) {
    db_x0           .push_back(tree.GetVal(0)[i]);
    db_y0           .push_back(tree.GetVal(1)[i]);
    db_z0           .push_back(tree.GetVal(2)[i]);
    db_barrel_endcap.push_back(tree.GetVal(3)[i]);
    db_layer_disk   .push_back(tree.GetVal(4)[i]);
    db_eta_module   .push_back(tree.GetVal(5)[i]);
    db_phi_module   .push_back(tree.GetVal(6)[i]);
  }
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <class T>
std::vector<T> strip_module_DB<T>::get_module_position (const int& it, const int& barrel_endcap, const int& layer_disk, const int& eta_module, const int& phi_module)
//-------------------------------------------------------------------------------------------------------------------------------------------------------------------
{
  if (_positions.find (it) == _positions.end()) {
    if (barrel_endcap) _positions.insert (std::pair<int, std::vector<T>> (it, std::vector<T> (3, -9999)));

    for (int i=0; i<nMods; i++)
      if (db_barrel_endcap[i] == barrel_endcap  &&  db_layer_disk[i] == layer_disk  &&  db_eta_module[i] == eta_module  &&  db_phi_module[i] == phi_module) {
        _positions.insert (std::pair <int, std::vector<T>> (it, std::vector<T> {db_x0[i], db_y0[i], db_z0[i]}));
        i = nMods;
      }
  }

  return (_positions.equal_range(it).first)->second;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <class T>
std::vector<T> strip_module_DB<T>::get_module_position (const int& it, const std::string & hardware, const int& barrel_endcap, const int& layer_disk, const int& eta_module, const int& phi_module)
//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
{
  std::vector<T> ret;
  ret.assign (3, -9999);

  if (hardware == 2) ret = get_module_position (it, barrel_endcap, layer_disk, eta_module, phi_module);

  return ret;
}

//--------------------------------------------------------------------------------------------------
template <class T>
template <class DT>
std::vector<T> strip_module_DB<T>::get_module_position (const int& it, const TTree_hits<DT>& TThits)
//--------------------------------------------------------------------------------------------------
{
  std::vector<T> ret;
  ret.assign (3, -9999);

  TThits.get (it);
  if (TThits.hardware() == 2)
   ret = get_module_position (it, TThits.barrel_endcap(), TThits.layer_disk(), TThits.eta_module(), TThits.phi_module());

  return ret;
}

