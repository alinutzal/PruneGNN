/***************************************************************
 * Tracking project library - L2IT
 * Trace reconstruction in LHC
 * copyright © 2022,2023 ROUGIER Charline
 * copyright © 2022,2023 COLLARD Christophe
 * copyright © 2022,2023 Centre National de la Recherche Scientifique
 * copyright © 2022,2023 Laboratoire des 2 Infinis de Toulouse (L2IT)
 ***************************************************************/

#include "hits_infos.hpp"

//---------------------------------------------------------------------
template <class T>
hits_infos<T>::hits_infos (TTree_hits<T>& TTh, TTree_particles<T>& TTp)
//---------------------------------------------------------------------
{ assert (TTh.size() && TTp.size());

  // loop on particles (faster - no particles set to 0 like in hits)
  for (int pID=0; pID<TTp.size(); pID++){
    TTp.get(pID);
    uint64_t particleID = TTp.particle_ID();
    std::pair<std::_Rb_tree_iterator<std::pair<const uint64_t, int>>, std::_Rb_tree_iterator<std::pair<const uint64_t, int>>> HitRange = TTh.particleID_equal_range (particleID);
    std::multimap<uint64_t, std::pair<int, T>> moduleID_HitRange_map; // moduleID / hit nb / norm of (x_v, y_v, z_v)
    std::multimap<T, int> norm_Hit_map;
    for (std::_Rb_tree_iterator<std::pair<const uint64_t, int>> hit = HitRange.first; hit != HitRange.second; hit++) {
      TTh.get(hit->second);
      // sort HitRange by moduleID
      T x_v = TTh.x() - TTp.vx();
      T y_v = TTh.y() - TTp.vy();
      T z_v = TTh.z() - TTp.vz();
      T norm = std::sqrt (x_v * x_v + y_v * y_v + z_v * z_v);
      moduleID_HitRange_map.insert (std::pair<uint64_t, std::pair<int, T>> (TTh.module_ID(), std::pair<int, T> (hit->second, norm)));
    }

    for (std::_Rb_tree_iterator<std::pair<const uint64_t, std::pair<int, T>>> hit = moduleID_HitRange_map.begin(); hit != moduleID_HitRange_map.end();){
      std::_Rb_tree_iterator<std::pair<const uint64_t, std::pair<int, T>>> next_hit = hit;
      next_hit++;

      if (next_hit == moduleID_HitRange_map.end()) {
        norm_Hit_map.insert (std::pair<T, int> (hit->second.second, hit->second.first));
        hit++;
      }
      else {
        assert (next_hit != hit);
        if (hit->first == next_hit->first)
          if (hit->second.second > next_hit->second.second)
            moduleID_HitRange_map.erase (next_hit);
          else {
            moduleID_HitRange_map.erase (hit);
            hit = next_hit;
          }
        else {
          norm_Hit_map.insert (std::pair<T, int> (hit->second.second, hit->second.first));
          hit++;
    } } }

    for (std::pair<T, int> hit : norm_Hit_map)
      _particleID_TTreeHits_map.insert (std::pair<uint64_t, int> (particleID, hit.second));
  }

  _size = _particleID_TTreeHits_map.size();
}
