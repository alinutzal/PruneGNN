/***************************************************************
 * Tracking project library - L2IT
 * Trace reconstruction in LHC
 * copyright © 2022,2023 ROUGIER Charline
 * copyright © 2022,2023 COLLARD Christophe
 * copyright © 2022,2023 Centre National de la Recherche Scientifique
 * copyright © 2022,2023 Laboratoire des 2 Infinis de Toulouse (L2IT)
 ***************************************************************/

#ifndef __cplusplus
#error Must use C++ for the type hits infos
#endif

#include <iostream>
#include <fstream>
#include <assert.h>
#include <vector>
#include <TTree_hits>
#include <TTree_particles>


//=====================================================================
template <class T> class hits_infos : TTree_hits<T>, TTree_particles<T>
//=====================================================================
{
//  friend hits;

  private:
    int _size;
    int _position;
//    std::vector<T> r;
//    std::vector<int> particle;
//    vector<T> _x_v, _y_v, _z_v;
    std::multimap<uint64_t, int> _particleID_TTreeHits_map;

    // particles properties for std root files compatibility only

  public:
    hits_infos ();
    hits_infos (TTree_hits<T>&, TTree_particles<T>&);

    inline int size () {return _size;}
    inline std::pair<std::_Rb_tree_iterator<std::pair<const uint64_t, int>>, std::_Rb_tree_iterator<std::pair<const uint64_t, int>>> particleID_equal_range (const uint64_t& ID) {return _particleID_TTreeHits_map.equal_range(ID);}

};
