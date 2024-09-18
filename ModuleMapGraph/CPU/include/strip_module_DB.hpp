/***************************************************************
 * Tracking project library - L2IT
 * Trace reconstruction in LHC
 * copyright © 2023 TORRES Heberth
 * copyright © 2023 Centre National de la Recherche Scientifique
 * copyright © 2023 Laboratoire des 2 Infinis de Toulouse (L2IT)
 ***************************************************************/

#ifndef __cplusplus
#error Must use C++ for the type strip module DB
#endif

#include <iostream>
#include <TString.h>
#include <TChain.h>
#include <vector>
#include <TTree_hits>

//======================================
template <class T> class strip_module_DB
//======================================
{
  private:
    int nMods;
    std::vector<T> db_x0;
    std::vector<T> db_y0;
    std::vector<T> db_z0;
    std::vector<char> db_barrel_endcap;
    std::vector<char> db_layer_disk;
    std::vector<char> db_eta_module;
    std::vector<char> db_phi_module;
    std::multimap <int, std::vector<T>> _positions;

  public:
    strip_module_DB () {}
    strip_module_DB (const std::string&);
    std::vector<T> get_module_position (const int&, const int&, const int&, const int&, const int&);
    std::vector<T> get_module_position (const int&, const std::string&, const int&, const int&, const int&, const int&);
    template <class DT> std::vector<T> get_module_position (const int&, const TTree_hits<DT>&);
};
