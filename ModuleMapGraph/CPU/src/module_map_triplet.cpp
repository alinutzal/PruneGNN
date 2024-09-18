/***************************************************************
 * Tracking project library - L2IT
 * Trace reconstruction in LHC
 * copyright © 2022,2023 ROUGIER Charline
 * copyright © 2022,2023,2024 COLLARD Christophe
 * copyright © 2022,2023,2024 Centre National de la Recherche Scientifique
 * copyright © 2022,2023,2024 Laboratoire des 2 Infinis de Toulouse (L2IT)
 ***************************************************************/

#include "module_map_triplet.hpp"


//=====Private methods for module map triplet===========================================


//=====Public methods for module map triplet============================================


//-----------------------------------------------------------------------------------------------------------------------------------------------------------------
template <class T>
module_map_triplet<T>::module_map_triplet (boost::program_options::variables_map& po_vm, const std::string& hits_filenames, const std::string& particles_filenames)
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------
{
  std::string events_path = po_vm["input-dir"].as<std::string> () + "/";
//  output_dir = po_vm["output-dir"].as<std::string> ();
//  true_graph = po_vm["give-true-graph"].as<bool> ();
  std::string module_map_dir = po_vm["input-module-map"].template as<std::string> ();
//  bool not_follow_electrons = po_vm["no-follow-electron"].as<bool> ();
  //  save_graph = po_vm["save-graph-on-disk"].as<bool> ();
  int min_N_hits = po_vm["min-nhits"].as<long unsigned int> ();
  T min_pt = po_vm["min-pt-cut"].as<T> ();
  T max_pt = po_vm["max-pt-cut"].as<T> ();
//  _phi_slice = po_vm["phi-slice"].as<bool> ();
//  _phi_slice_cut1 = po_vm["cut1-phi-slice"].as<T> ();
//  _phi_slice_cut2 = po_vm["cut2-phi-slice"].as<T> ();
//  _eta_region = po_vm["eta-region"].as<bool> ();
//  _eta_cut1 = po_vm["cut1-eta"].as<T> ();
//  _eta_cut2 = po_vm["cut2-eta"].as<T> ();
  _strip_hit_pair = po_vm["strip-hit-pair"].as<bool> ();

//  if (module_map_dir.empty()) throw std::invalid_argument ("Missing Module Map");

//  m_modMapTriplet.read_TTree (module_map_dir.c_str());


/*
  std::string module_map_dir_output = "/sps/l2it/collard/MMTriplet_3hits_ptCut300MeV_events_0kto1_woutOverlapSP_particleInfo.out.root";
  m_modMapTriplet.save_TTree (module_map_dir_output);
  module_map_triplet<T> m_modMap2;
  m_modMap2.read_TTree (module_map_dir_output);
  if (m_modMapTriplet == m_modMap2) std::cout << "OK !!!!! YUPI" << std::endl;
*/

//  if (!m_modMapTriplet) throw std::runtime_error ("Cannot retrieve ModuleMap from " + module_map_dir);
//  event_id = "";

  clock_t start, end;
  clock_t start_main, end_main;

  start_main = start = clock();

  // open files containing events filenames
  std::ifstream h_file (hits_filenames);
  if (h_file.fail()) throw std::invalid_argument ("Cannot open file " + hits_filenames);
  std::ifstream p_file (particles_filenames);
  if (p_file.fail()) throw std::invalid_argument ("Cannot open file " + particles_filenames);

  std::string hits_filename, particles_filename;
  h_file >> hits_filename;
  p_file >> particles_filename;

  for (; !h_file.eof() && !p_file.eof();) {
    std::string event_id = boost::regex_replace (hits_filename, boost::regex("[^0-9]*([0-9]+).*"), std::string("\\1"));
    if (event_id != boost::regex_replace (particles_filename, boost::regex("[^0-9]*([0-9]+).*"), std::string("\\1")))
      throw std::invalid_argument ("hits and particules file are not part of the same event");

    add_event_triplets (events_path, hits_filename, particles_filename, min_N_hits, min_pt, max_pt);
    h_file >> hits_filename;
    p_file >> particles_filename;
  }

  h_file.close();
  p_file.close();

  // map doublets and modules
  int loop = 0;
  for (std::pair<std::vector<uint64_t>, module_doublet<T>> ModuleDoublet : _module_map_doublet) {
    _module_map_pairs.insert (std::pair<std::vector<uint64_t>, int> (ModuleDoublet.first, loop));
    loop++;
    if (_module_map.find(ModuleDoublet.first[0]) == _module_map.end()) _module_map.insert (std::pair<uint64_t, int> (ModuleDoublet.first[0],0));
    if (_module_map.find(ModuleDoublet.first[1]) == _module_map.end()) _module_map.insert (std::pair<uint64_t, int> (ModuleDoublet.first[1],0));
  }

  loop=0;
  for (std::multimap<uint64_t,int>::iterator it= _module_map.begin(); it != _module_map.end(); ++it, loop++) {
    it->second = loop;
    loop++;
  }
  std::cout << green << "# modules in MM = " << _module_map.size() << reset;
}


//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <class T>
void module_map_triplet<T>::add_event_triplets (const std::string& path, const std::string& hits_filename, const std::string& particles_filename, const int& min_N_hits, const T& min_pt, const T& max_pt)
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
{
  // load event hits
  hits<T> hit_pts;  // define in class data ?
  TTree_hits<T> TThits;
  std::filesystem::path hits_path = hits_filename;
  if (hits_path.extension () == ".csv")
    { hit_pts.read_csv (path, hits_filename);
      TThits = hit_pts;
    }
  else if (hits_path.extension () == ".root")
    TThits.read (path + hits_filename);
  else throw std::invalid_argument ("unknow extension for hit file" + hits_filename);

  // load event particles
  particles<T> p;
  TTree_particles<T> TTp;
  std::filesystem::path particles_path = particles_filename;
  if (particles_path.extension () == ".csv")
    { p.read_csv (path, particles_filename);
      TTp = p;
    }
  else if (particles_path.extension () == ".root")
    TTp.read (path + particles_filename);
  else throw std::invalid_argument ("unkonow extension for particles file" + particles_filename);

//  using HitGraphContainer = boost::unordered::iterator_detail::iterator<boost::unordered::detail::ptr_node <std::pair<const uint64_t, int> > >;
  using HitGraphContainer = boost::unordered::unordered_map<long unsigned int, int>::iterator;

  hits_infos<T> HitsInfos (TThits, TTp);

  // loop on particles (faster - no particles set to 0 like in hits)
  for (int pID=0; pID<TTp.size(); pID++){
    TTp.get (pID);
    uint64_t particleID = TTp.particle_ID();
//    T pT = 0.001 * TTp.pT();
//    std::cout << "PID = " << TTp.particle_ID() << std::endl;

    //----------
    // cut on pt
    //----------
    if (0.001*TTp.pT() < min_pt) continue;
    if (max_pt && 0.001*TTp.pT() > max_pt) continue;

    //------------------------------------
    // cut on barcode (ignore secondaries)
    //------------------------------------
    if (TTp.barcode() > 200000) continue;

    //--------------------------------
    // Do we want to follow electron ?
    //--------------------------------
//    if (not_follow_electrons)
      if (abs (TTp.pdgID()) == 11) continue;

    //-------------------------------------------------
    // Preselection on number of hits (at least 2 hits)
    //-------------------------------------------------
//    std::pair<std::_Rb_tree_iterator<std::pair<const uint64_t, int>>, std::_Rb_tree_iterator<std::pair<const uint64_t, int>>> HitRange = TThits.particleID_equal_range (particleID);
    std::pair<std::_Rb_tree_iterator<std::pair<const uint64_t, int>>, std::_Rb_tree_iterator<std::pair<const uint64_t, int>>> HitRange = HitsInfos.particleID_equal_range (particleID); // test on HitsInfo (should be faster)
    int size = distance (HitRange.first, HitRange.second);
    if (size < min_N_hits) continue; //remove particles leaving less than min_N_hits here

    //-------------------------
    // Split cluster management
    //-------------------------
    // it happens that two hits (or even more) belonging to the same particle are on the same module (i.e split clusters)
    // in that case, as the module map connect a module to another one, one of the hit is removed (the second)

    // sort the hits so that hit belonging to the same module are next to each other in the vector
    if (abs (TTp.eta ()) > 4) continue;

    if (std::sqrt(TTp.vx()*TTp.vx() + TTp.vy()*TTp.vy()) > 260) continue;

    //--------------------
    // Module Map creation
    //--------------------

    for (std::_Rb_tree_iterator<std::pair<const uint64_t, int>> hit = HitRange.first; hit != HitRange.second; hit++){
      int it1 = hit->second;
      std::_Rb_tree_iterator<std::pair<const uint64_t, int>> next_hit, next_next_hit;
      next_next_hit = next_hit = hit;
      int it2 = (++next_hit)->second;
      if (next_hit == HitRange.second) break;

      int it3 = (++(++next_next_hit))->second;
      if (next_next_hit == HitRange.second) break;

      uint64_t module1 = TThits.module_ID(it1);
      uint64_t module2 = TThits.module_ID(it2);
      uint64_t module3 = TThits.module_ID(it3);
      const std::vector<uint64_t> module = {module1, module2, module3};

      geometric_cuts<T> cuts12 (TThits, it1, it2);
      geometric_cuts<T> cuts23 (TThits, it2, it3);

      cuts12.check_cut_values ();
      cuts23.check_cut_values ();

      T diff_dzdr = Diff_dzdr (TThits, it1, it2, it3);
      T diff_dydx = Diff_dydx (TThits, it1, it2, it3);

//      if (isnan (diff_dzdr))
//          std::cout << "Error: diff_dzdr Not A Number, you cannot add the link to the module map." << std::endl;

//      if (isnan (diff_dydx))
//        std::cout << "Error: diff_dydx Not A Number, you cannot add the link to the module map." << std::endl;

//      if (isinf(diff_dzdr))
//        std::cout << "Error: diff_dzdr is infinite, you cannot add the link to the module map." << std::endl;

//      if (isinf(diff_dydx))
//        std::cout << "Error: diff_dydx is infinite, you cannot add the link to the module map." << std::endl;

//      add_triplet_occurence (module, TThits, TTp, cuts12, cuts23, diff_dzdr, diff_dydx);
      add_triplet_occurence (module, TThits, TTp, cuts12, cuts23, diff_dzdr, diff_dydx, it1, it2, it3);
    }
  }
}


/*
//--------------------------------------------
template <class Tf>
bool operator += (const module_triplet<T>& mt)
//--------------------------------------------
{
  if( isnan(mt.modules12().z0_12) || isnan(z0_23)){
        std::cout<<"Error : z0 Not A Number , you cannot add the link to the module map."<<std::endl;
        return false;
    }
    if( isnan(dphi_12) || isnan(dphi_23)){
        std::cout<<"Error : dphi Not A Number , you cannot add the link to the module map."<<std::endl;
        return false;
    }
    if( isnan(phiSlope_12) || isnan(phiSlope_23) ){
        std::cout<<"Error : phiSlope Not A Number , you cannot add the link to the module map."<<std::endl;
        return false;
    }
    if( isnan(deta_12) || isnan(deta_23) ){
        std::cout<<"Error : deta Not A Number , you cannot add the link to the module map."<<std::endl;
        return false;
    }

    if( isnan(diff_dydx)){
        std::cout<<"Error : diff_dydx Not A Number , you cannot add the link to the module map."<<std::endl;
        return false;
    }

    if( isnan(diff_dzdr)){
        std::cout<<"Error : diff_dzdr Not A Number , you cannot add the link to the module map."<<std::endl;
        return false;
    }

    if( isinf(z0_12) || isinf(z0_23) ){
        std::cout<<"Error : z0 is infinite , you cannot add the link to the module map. "<<std::endl;
        return false;
    }
    if( isinf(dphi_12) || isinf(dphi_23)){
        std::cout<<"Error : dphi is infinite , you cannot add the link to the module map."<<std::endl;
        return false;
    }
    if( isinf(phiSlope_12) || isinf(phiSlope_23)){
        std::cout<<"Error : phiSlope is infinite , you cannot add the link to the module map."<<std::endl;
        return false;
    }
    if( isinf(deta_12) || isinf(deta_23) ){
        std::cout<<"Error : deta is infinite , you cannot add the link to the module map."<<std::endl;
        return false;
    }

    if( isinf(diff_dzdr) ){
        std::cout<<"Error : diff_dzdr is infinite , you cannot add the link to the module map."<<std::endl;
        return false;
    }

    if( isinf(diff_dydx)){
        std::cout<<"Error : diff_dydx is infinite , you cannot add the link to the module map."<<std::endl;
        return false;
    }
}
*/


//---------------------------------------------------------------------------------------
template <class Tf>
bool operator == (const module_map_triplet<Tf>& mmt1, const module_map_triplet<Tf>& mmt2)
//---------------------------------------------------------------------------------------
{
  return (mmt1._module_map_triplet == mmt2._module_map_triplet);
}


//---------------------------------------------------------------------------------------
template <class Tf>
bool operator != (const module_map_triplet<Tf>& mmt1, const module_map_triplet<Tf>& mmt2)
//---------------------------------------------------------------------------------------
{
  return !(mmt1 == mmt2);
}


//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <class T>
//bool module_map_triplet<T>::add_triplet_occurence (const std::vector<uint64_t>& triplet, const TTree_hits<T>& TThits, const TTree_particles<T>& TTp, const geometric_cuts<T>& cuts12, const geometric_cuts<T>& cuts23, const T& diff_dzdr, const T& diff_dydx)
bool module_map_triplet<T>::add_triplet_occurence (const std::vector<uint64_t>& triplet, const TTree_hits<T>& TThits, const TTree_particles<T>& TTp, const geometric_cuts<T>& cuts12, const geometric_cuts<T>& cuts23, const T& diff_dzdr, const T& diff_dydx, int it1, int it2, int it3)
//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
{
  bool new_particle = true;

//  int nb_particles = p.size();
//  std::pair <std::_Rb_tree_iterator <std::pair <const std::vector<uint64_t>, module_triplet<T, U>>>, std::_Rb_tree_iterator <std::pair <const std::vector<uint64_t>, module_triplet<T, U>>>> module_triplets = _module_map_triplet.equal_range (triplet);
//  std::_Rb_tree_iterator <std::pair <const std::vector<uint64_t>, module_triplet<T, U>>> mod_triplet = _module_map_triplet.find (triplet);
//  module_triplet<T, U> mod_triplet = _module_map_triplet.find (triplet)->second;
///  std::vector<int> particles_indices = _module_particles.find (triplet)->second;
////  std::pair <std::_Rb_tree_iterator<std::pair <const std::vector<uint64_t>, int>>, std::_Rb_tree_iterator <std::pair <const std::vector<uint64_t>, int>>>  mod_triplet = _modules_particles.equal_range (triplet);
  std::pair <std::_Rb_tree_iterator<std::pair <const std::vector<uint64_t>, std::vector<T>>>, std::_Rb_tree_iterator <std::pair <const std::vector<uint64_t>, std::vector<T>>>>  mod_triplet = _modules_particlesKey.equal_range (triplet);

///  std::_Rb_tree_iterator<std::pair <const std::vector<uint64_t>, int>> mod_triplet = _module_map_triplet.find (triplet);
  int size = distance (mod_triplet.first, mod_triplet.second);

//  if (mod_triplet == _module_map_triplet.end()) {
//    _module_map_triplet.insert (std::pair <std::vector<uint64_t>, module_triplet<T,U>> (triplet, module_triplet<T,U> (cuts12, cuts23, Diff_dzdr (TThits, triplet[0], triplet[1], triplet[2]), Diff_dydx (TThits, triplet[0], triplet[1], triplet[2])));
//  }
  std::vector<T> key;// = TTp.key();
  key.push_back (TTp.pT());
  key.push_back (TTp.pdgID());
  key.push_back (TTp.eta());
  key.push_back (TTp.vx());
  key.push_back (TTp.vy());
  key.push_back (TTp.vz());

  for (std::_Rb_tree_iterator<std::pair <const std::vector<uint64_t>, std::vector<T>>> it = mod_triplet.first; it != mod_triplet.second && new_particle; it++)
    new_particle *= (it -> second != key);

  if (new_particle) {
    _modules_particlesKey.insert (std::pair <std::vector<uint64_t>, std::vector<T>> (triplet, key));
    if (_particlesSPECS_TTp_link.find (key) == _particlesSPECS_TTp_link.end()) {
      _particlesSPECS_TTp_link.insert (std::pair <std::vector<T>, int> (key, _ModMap_particles.size()));
    }
////  else 
////      _modules_particles.insert (std::pair <const std::vector<uint64_t>, int> (triplet, _particlesSPECS_TTp_link.find (key)->second));
//    _module_map_triplet.set (cuts12, cuts23, Diff_dzdr (TThits, triplet[0], triplet[1], triplet[2]), Diff_dydx (TThits, triplet[0], triplet[1], triplet[2]));
    _cuts12.push_back (cuts12);
    _cuts23.push_back (cuts23);
  }

  std::vector<uint64_t> doublet12 {triplet.at(0), triplet.at(1)};
  std::vector<uint64_t> doublet23 {triplet.at(1), triplet.at(2)};

  if (!size) {
    if (_module_map_doublet.find (doublet12) == _module_map_doublet.end())
      _module_map_doublet.insert (std::pair <std::vector<uint64_t>, module_doublet<T>> (doublet12, module_doublet<T> (cuts12)));
    else
      (_module_map_doublet.find (doublet12) -> second).add_occurence (cuts12);
    if (_module_map_doublet.find (doublet23) == _module_map_doublet.end())
      _module_map_doublet.insert (std::pair <std::vector<uint64_t>, module_doublet<T>> (doublet23, module_doublet<T> (cuts23)));
    else
      (_module_map_doublet.find (doublet23) -> second).add_occurence (cuts23);
    _module_map_triplet.insert (std::pair <std::vector<uint64_t>, module_triplet<T>> (triplet, module_triplet<T> (cuts12, cuts23, diff_dzdr, diff_dydx)));
//    if (triplet == std::vector<uint64_t> {1460151441686528, 19764821020901376, 19769219067412480}) {
//      std::cout << "barcode = " << TTp.barcode () << " / subevent = " << TTp.subevent() << " / event ID = " << TTp.event_ID() <<  " / TTp particle ID = " << TTp.particle_ID() << std::endl;
//      std::cout << "hits 1: id = " << TThits.hit_id(it1) << " / x = " << TThits.x(it1) << " / y = " << TThits.y(it1) << " / z = " << TThits.z(it1) << std::endl;
//      std::cout << "hits 2: id = " << TThits.hit_id(it2) << " / x = " << TThits.x(it2) << " / y = " << TThits.y(it2) << " / z = " << TThits.z(it2) << std::endl;
//      std::cout << "hits 3: id = " << TThits.hit_id(it3) << " / x = " << TThits.x(it3) << " / y = " << TThits.y(it3) << " / z = " << TThits.z(it3) << std::endl;
//    }
  }
  else {
    (_module_map_doublet.find (doublet12) -> second).add_occurence (cuts12);
    (_module_map_doublet.find (doublet23) -> second).add_occurence (cuts23);
    (_module_map_triplet.find (triplet) -> second).add_occurence (cuts12, cuts23, diff_dzdr, diff_dydx);
//    if (triplet == std::vector<uint64_t> {1460151441686528, 19764821020901376, 19769219067412480}) {
//      std::cout << "modif: barcode = " << TTp.barcode () << " / subevent = " << TTp.subevent() << " / event ID = " << TTp.event_ID() <<  " / TTp particle ID = " << TTp.particle_ID() << std::endl;
//      std::cout << "modif hits 1: id = " << TThits.hit_id(it1) << " / x = " << TThits.x(it1) << " / y = " << TThits.y(it1) << " / z = " << TThits.z(it1) << std::endl;
//      std::cout << "modif hits 2: id = " << TThits.hit_id(it2) << " / x = " << TThits.x(it2) << " / y = " << TThits.y(it2) << " / z = " << TThits.z(it2) << std::endl;
//      std::cout << "modif hits 3: id = " << TThits.hit_id(it3) << " / x = " << TThits.x(it3) << " / y = " << TThits.y(it3) << " / z = " << TThits.z(it3) << std::endl;
//    }
  }


//    particles_indices.push_back(_ModMap_particles.size()); // is not gonna work -> add vector apart
    // too slow - replace by a table of std::multimap <std::vector<uint64_t>, int> _module_particles;
    // no need to copy, delete and insert => faster
//    _module_particles.erase (triplet);
//    _module_particles.insert (std::pair <std::vector<uint64_t>, std::vector<int>> (triplet, particles_indices));
//  }

//       std::cout << "Modules : (" << triplet[0] << ", " << triplet[1] << ", " << triplet[2] << ") \n";
//       std::cout << (_module_map_triplet.find (triplet) -> second).modules23().z0_min() << " / " << (_module_map_triplet.find (triplet) -> second).modules23().z0_max() << std::endl;

//  std::cout << "PID = " << TTp.particle_ID() << std::endl;
//x    if (abs (cuts12.d_phi() + 0.00243449) < 1E-8 || abs (cuts23.d_phi() + 0.00243449) < 1E-8) {
//x      std::cout << "Module : (" << triplet[0] << ", " << triplet[1] << ", " << triplet[2] << ") \n";
//x      exit(0);
//x    }
//if (triplet[0] == 414612640694796288 && triplet[1] == 455162629527175168 && triplet[2] == 1446728603734638592 && temp_eventID == 5) exit(0);
//  if (triplet[0] == 114463558497992704 && triplet[1] == 151644643702865920 && triplet[2] == 1341914359282008064) {
//x  if (triplet[0] == 293367294476681216 && triplet[1] == 158329674399744 && triplet[2] == 167125767421952) {
//x    std::cout << cuts12.d_phi() << " / " << cuts23.d_phi() << std::endl;
//x    exit(0);
//x  }
//    if (abs ((_module_map_triplet.find (triplet) -> second).modules12().z0_min() + 39.7335) < 1E-4) exit(0);
//  if (triplet[0] == 296278801267032064 && triplet[1] == 295724647406632960 && triplet[2] == 334014040332304384) exit(0);
//  if (triplet[0] == 8162774324609024 && triplet[1] == 7608620464209920 && triplet[2] == 8171570417631232) exit(0);
//  if (triplet[0] == 290675690011885568 && triplet[1] == 327276233077293056 && triplet[2] == 365020268235587584) exit(0);

  return new_particle;
}


//-------------------------------------------------------------------------------
template <class T>
void module_map_triplet<T>::merge (const module_map_triplet<T>& ModuleMapTriplet)
//-------------------------------------------------------------------------------
{
  // Loop on all entries of the module map in input
  for (const std::pair<std::vector<uint64_t>, module_triplet<T>> MM_entry : ModuleMapTriplet._module_map_triplet)
    if (_module_map_triplet.find (MM_entry.first) == _module_map_triplet.end())
      _module_map_triplet.insert (std::pair <std::vector<uint64_t>, module_triplet<T>> (MM_entry.first, MM_entry.second));
    else
      (_module_map_triplet.find (MM_entry.first) -> second).add_occurence (MM_entry.second);

//      _modules_particlesKey.insert (std::pair <std::vector<uint64_t>, std::vector<T>> (MM_entry.first, ModuleMapTriplet.particles_key(MM_entry.first))); // probably useless ?
//      _modules_particles.insert (std::pair <std::vector<uint64_t>, std::vector<int>> (MM_entry.first, MM_entry.second));
//
//      bool new_particle = true;
//      std::vector<int> particles = _modules_particles.find (MM_entry.first).second;
//      std::vector<int> new_particles = ModuleMap_triplet.particles_position (MM_entry.first);
//      for (int i=0; i<new_particles.size() && new_particle; i++) {
//        vector<int> new_key = TTp->key (new_particles_position[i]);
//        for (int j=0; j<particles.size() && new_particle; j++)
//          new_particle = (new_key == TTp.key (_modules_particles.key [j]));
//
//      if (new_particle) {
//        _modules_particles.find (MM_entry.first).second.push_back ();

  for (const std::pair<std::vector<uint64_t>, module_doublet<T>> MM_entry : ModuleMapTriplet._module_map_doublet) {
    if (_module_map_doublet.find (MM_entry.first) == _module_map_doublet.end())
      _module_map_doublet.insert (std::pair <std::vector<uint64_t>, module_doublet<T>> (MM_entry.first, MM_entry.second));
    else
      (_module_map_doublet.find (MM_entry.first) -> second).add_occurence (MM_entry.second);
  }

//    add_triplet_occurence (MM_entry.first, MM_entry.second->modules12().)

//    std::vector<uint64_t> triplet = MM_entry.first;
//    if (_module_map_triplet.find (triplet) == _module_map_triplet.end()) {
//    }
//  }
//  for (const std::pair<std::vector<uint64_t>, module_triplet<T,U>> MM_entry : ModuleMapTriplet._module_map_triplet)
//    add_triplet_occurence (MM_entry.second, MM_entry._cuts12, MM_entry._cuts23, const T& diff_dzdr, const T& diff_dydx);
}


//---------------------------------
template <class T>
void module_map_triplet<T>::sort ()
//---------------------------------
{
  std::multimap <std::vector<uint64_t>, unsigned int> ModulesTriplets_to_keep;
  std::multimap <std::vector<uint64_t>, module_triplet<T>> sorted_module_map_triplet = _module_map_triplet;
int step = 0;

  for (std::pair<std::vector<uint64_t>, module_triplet<T>> MM_entry : _module_map_triplet) {
    bool unique = (MM_entry.second.occurence() == 1) || (MM_entry.second.min_equal_max());
    uint64_t module1 = MM_entry.first[0];
    uint64_t module2 = MM_entry.first[1];
    uint64_t module3 = MM_entry.first[2];
    bool add = true;

    for (std::pair<std::vector<uint64_t>, unsigned int> modules : ModulesTriplets_to_keep) {
      uint64_t m1 = modules.first[0];
      uint64_t m2 = modules.first[1];
      uint64_t m3 = modules.first[2];
      unsigned int unique2 = modules.second;
      if (unique2 > 1) std::cout << "unique2 = "  << unique2 << std::endl;

//      if (module1 == m3) && ((module2 == m1 || module2 == m2 || )
      if (((m1 == module2 && m2 == module1)  ||  (m2 == module2) && (m1 == module3)) ||
          ((m3 == module2 && m2 == module3)  ||  (m2 == module2) && (m3 == module1)) ||
          ((m3 == module1 && m1 == module2)  ||  (m3 == module2) && (m1 == module3))) {
            add = false;
            if (unique2 && !unique) sorted_module_map_triplet.erase ({m1, m2, m3});
            else if (unique && !unique2) sorted_module_map_triplet.erase ({module1, module2, module3});
            else if (unique > unique2) sorted_module_map_triplet.erase ({m1, m2, m3});
            else sorted_module_map_triplet.erase ({module1, module2, module3});
      }
    }

//    if (add) ModulesTriplets_to_keep.insert (std::pair<std::vector<uint64_t>, unsigned int> ({module1, module2, module3}, MM_entry.second.occurence()));
    if (add) ModulesTriplets_to_keep.insert (std::pair<std::vector<uint64_t>, unsigned int> ({module1, module2, module3}, unique));
//      std::cout << "step = " << ++step << std::endl;
  }
  _module_map_triplet = sorted_module_map_triplet;
}


/*
//-----------------------------------
template <class T>
void module_map_triplet<T>::remove_cycles()
//-----------------------------------
{
  std::multimap <std::vector<uint64_t>, unsigned int> ModulesTriplets_to_keep;
  std::multimap <std::vector<uint64_t>, module_triplet<T>> sorted_module_map_triplet = _module_map_triplet;

  // Loop on all initial MM entries
  // ------------------------------
  for (std::pair<std::vector<uint64_t>, module_triplet<T>> MM_entry : _module_map_triplet) {

    // Force occurence to 1 if for some cuts min==max (same particle was seen several times, making occurence > 1)
    unsigned int occurence_unique = (MM_entry.second.min_equal_max())? 1 : MM_entry.second.occurence();
    uint64_t module1 = MM_entry.first[0];
    uint64_t module2 = MM_entry.first[1];
    uint64_t module3 = MM_entry.first[2];
    bool add = true;

    // Loop on all triplets already kept
    // if one cycle will appear, we remove the triplet with less occurence (or only one)
    for (std::pair<std::vector<uint64_t>, unsigned int> modules : ModulesTriplets_to_keep) {


      uint64_t m1 = modules.first[0];
      uint64_t m2 = modules.first[1];
      uint64_t m3 = modules.first[2];
      unsigned int occurence_unique2 = modules.second;

      //if (unique2 > 1) std::cout << "unique2 = "  << unique2 << std::endl;

      // //CHarline's if
      // if ( ( m1 == module2 && m2 == module1)  || ( m2 == module2 && m1 == module3) ){
      // else if ((m3 == module1 && m2 == module2) || (m3==module2 && m2 == module3)){
      // else if ((m3 == module1 && m1 == module2) || (m3 == module2 && m1 == module3) ){

      if (((m1 == module2 && m2 == module1)  ||  (m2 == module2) && (m1 == module3)) ||
          ((m3 == module2 && m2 == module3)  ||  (m2 == module2) && (m3 == module1)) ||
          ((m3 == module1 && m1 == module2)  ||  (m3 == module2) && (m1 == module3))) {
            add = false;
            if (occurence_unique2==1 && occurence_unique>1) sorted_module_map_triplet.erase ({m1, m2, m3});
            else if (occurence_unique==1 && occurence_unique2>1) sorted_module_map_triplet.erase ({module1, module2, module3});
            else if (occurence_unique > occurence_unique2) sorted_module_map_triplet.erase ({m1, m2, m3});
            else sorted_module_map_triplet.erase ({module1, module2, module3});
      }
    }

    if (add) ModulesTriplets_to_keep.insert (std::pair<std::vector<uint64_t>, unsigned int> ({module1, module2, module3}, occurence_unique));

  }
  _module_map_triplet = sorted_module_map_triplet;
}
*/
//
//----------------------------------------------------
template <class T>
void module_map_triplet<T>::remove_unique_occurence ()
//----------------------------------------------------
{
  int deleted = 0;
  for (std::_Rb_tree_iterator<std::pair<const std::vector<uint64_t>, module_triplet<T>>> it=_module_map_triplet.begin(); it!=_module_map_triplet.end(); it++) {
    unsigned int occurence = (it->second.min_equal_max()) ? 1 : it->second.occurence();
    uint64_t module1 = it->first[0];
    uint64_t module2 = it->first[1];
    uint64_t module3 = it->first[2];
    if (occurence == 1) {
      it--;
      _module_map_triplet.erase ({module1, module2, module3});
      deleted++;
    }
  }
  std::cout << "nb deleted triplets = " << deleted << std::endl;
}


//------------------------------------------
template <class T>
void module_map_triplet<T>::remove_cycles ()
//------------------------------------------
{
//  std::multimap <std::vector<uint64_t>, unsigned int> ModulesTriplets_to_keep;
//  std::multimap <std::vector<uint64_t>, module_triplet<T>> sorted_module_map_triplet = _module_map_triplet;
//  std::multimap <std::vector<uint64_t>, module_triplet<T>> sorted_module_map_triplet;
int count = 0;

  // Loop on all initial MM entries
  // ------------------------------
  for (std::_Rb_tree_iterator<std::pair<const std::vector<uint64_t>, module_triplet<T>>> it=_module_map_triplet.begin(); it!=_module_map_triplet.end(); it++) {
    unsigned int occurence_unique = (it->second.min_equal_max()) ? 1 : it->second.occurence();
    uint64_t module1 = it->first[0];
    uint64_t module2 = it->first[1];
    uint64_t module3 = it->first[2];
//    bool add = true;
      count++;
    if (count == 10000) std::cout << "10k" << std::endl;
    if (count == 100000) std::cout << "100k" << std::endl;
    if (count == 200000) std::cout << "200k" << std::endl;
    if (count == 500000) std::cout << "500k" << std::endl;
    if (count == 1000000) std::cout << "1M" << std::endl;

    for (std::_Rb_tree_iterator<std::pair<const std::vector<uint64_t>, module_triplet<T>>> it2=_module_map_triplet.begin(); it2!=it; it2++) {
      uint64_t m1 = it2->first[0];
      uint64_t m2 = it2->first[1];
      uint64_t m3 = it2->first[2];
      unsigned int occurence_unique2 = it2->second.occurence();
      std::vector<T> triplet_to_delete;

//  for (std::pair<std::vector<uint64_t>, module_triplet<T>> MM_entry : _module_map_triplet) {

    // Force occurence to 1 if for some cuts min==max (same particle was seen several times, making occurence > 1)
//    unsigned int occurence_unique = (MM_entry.second.min_equal_max())? 1 : MM_entry.second.occurence();
//    uint64_t module1 = MM_entry.first[0];
//    uint64_t module2 = MM_entry.first[1];
//    uint64_t module3 = MM_entry.first[2];
//    bool add = true;

    // Loop on all triplets already kept
    // if one cycle will appear, we remove the triplet with less occurence (or only one)
//    for (std::pair<std::vector<uint64_t>, unsigned int> modules : ModulesTriplets_to_keep) {


//      uint64_t m1 = modules.first[0];
//      uint64_t m2 = modules.first[1];
//      uint64_t m3 = modules.first[2];
//      unsigned int occurence_unique2 = modules.second;

      //if (unique2 > 1) std::cout << "unique2 = "  << unique2 << std::endl;

      // //CHarline's if
      // if ( ( m1 == module2 && m2 == module1)  || ( m2 == module2 && m1 == module3) ){
      // else if ((m3 == module1 && m2 == module2) || (m3==module2 && m2 == module3)){
      // else if ((m3 == module1 && m1 == module2) || (m3 == module2 && m1 == module3) ){

      if ((m1 == module2 && m2 == module1) || (m1 == module3 && m2 == module1) || (m1 == module3 && m2 == module2) ||
          (m1 == module2 && m3 == module1) || (m1 == module3 && m3 == module1) || (m1 == module3 && m3 == module2) ||
          (m2 == module2 && m3 == module1) || (m2 == module3 && m3 == module1) || (m2 == module3 && m3 == module2) ) {
        if (occurence_unique > occurence_unique2) {
          std::cout << "kill" << std::endl;
          it--; it2--;
          _module_map_triplet.erase ({m1, m2, m3});
        }
        else {
          it--;
          std::cout << "kill++" << std::endl;
          _module_map_triplet.erase ({module1, module2, module3});
          continue;
        }
      }
    }
  }

//    if (add) ModulesTriplets_to_keep.insert (std::pair<std::vector<uint64_t>, unsigned int> ({module1, module2, module3}, occurence_unique));

//  }
//  _module_map_triplet = sorted_module_map_triplet;
}

/*
//-----------------------------------------
template <class T>
void module_map_triplet<T>::remove_cycles()
//-----------------------------------------
{
  std::multimap <std::vector<uint64_t>, unsigned int> ModulesTriplets_to_keep;
  std::multimap <std::vector<uint64_t>, module_triplet<T>> sorted_module_map_triplet = _module_map_triplet;

  // Loop on all initial MM entries
  // ------------------------------
  for (std::pair<std::vector<uint64_t>, module_triplet<T>> MM_entry : _module_map_triplet) {

    // Force occurence to 1 if for some cuts min==max (same particle was seen several times, making occurence > 1)
    unsigned int occurence_unique = (MM_entry.second.min_equal_max())? 1 : MM_entry.second.occurence();
    uint64_t module1 = MM_entry.first[0];
    uint64_t module2 = MM_entry.first[1];
    uint64_t module3 = MM_entry.first[2];
    bool add = true;

    // Loop on all triplets already kept
    // if one cycle will appear, we remove the triplet with less occurence (or only one)
    for (std::pair<std::vector<uint64_t>, unsigned int> modules : ModulesTriplets_to_keep) {


      uint64_t m1 = modules.first[0];
      uint64_t m2 = modules.first[1];
      uint64_t m3 = modules.first[2];
      unsigned int occurence_unique2 = modules.second;

      //if (unique2 > 1) std::cout << "unique2 = "  << unique2 << std::endl;

      // //CHarline's if
      // if ( ( m1 == module2 && m2 == module1)  || ( m2 == module2 && m1 == module3) ){
      // else if ((m3 == module1 && m2 == module2) || (m3==module2 && m2 == module3)){
      // else if ((m3 == module1 && m1 == module2) || (m3 == module2 && m1 == module3) ){

      if (((m1 == module2 && m2 == module1)  ||  (m2 == module2) && (m1 == module3)) ||
          ((m3 == module2 && m2 == module3)  ||  (m2 == module2) && (m3 == module1)) ||
          ((m3 == module1 && m1 == module2)  ||  (m3 == module2) && (m1 == module3))) {
            add = false;
            if (occurence_unique2==1 && occurence_unique>1) sorted_module_map_triplet.erase ({m1, m2, m3});
            else if (occurence_unique==1 && occurence_unique2>1) sorted_module_map_triplet.erase ({module1, module2, module3});
            else if (occurence_unique > occurence_unique2) sorted_module_map_triplet.erase ({m1, m2, m3});
            else sorted_module_map_triplet.erase ({module1, module2, module3});
      }
    }

    if (add) ModulesTriplets_to_keep.insert (std::pair<std::vector<uint64_t>, unsigned int> ({module1, module2, module3}, occurence_unique));

  }
  _module_map_triplet = sorted_module_map_triplet;
}
*/


//-------------------------------------------------------------------------------
template <class T>
void module_map_triplet<T>::save_txt (std::string MM_output, const int precision)
//-------------------------------------------------------------------------------
{
  std::ofstream file ((MM_output+".triplets.txt").c_str(), std::ios::out);
  assert (!file.fail());
  for (std::pair <const std::vector<uint64_t>, module_triplet<T> >& MMentry : _module_map_triplet) {
    file << MMentry.first[0] << " " << MMentry.first[1] << " " << MMentry.first[2] << " ";
    file << MMentry.second.occurence() << " ";
    file << std::setprecision(precision) << MMentry.second.modules12().z0_max() << " " << std::setprecision(precision) << MMentry.second.modules12().z0_min() << " ";
    file << std::setprecision(precision) << MMentry.second.modules12().dphi_max() << " " << std::setprecision(precision) << MMentry.second.modules12().dphi_min() << " ";
    file << std::setprecision(precision) << MMentry.second.modules12().phi_slope_max() << " " << std::setprecision(precision) << MMentry.second.modules12().phi_slope_min() << " ";
    file << std::setprecision(precision) << MMentry.second.modules12().deta_max() << " " << std::setprecision(precision) << MMentry.second.modules12().deta_min() << " ";
    file << std::setprecision(precision) << MMentry.second.modules23().z0_max() << " " << std::setprecision(precision) << MMentry.second.modules23().z0_min() << " ";
    file << std::setprecision(precision) << MMentry.second.modules23().dphi_max() << " " << std::setprecision(precision) << MMentry.second.modules23().dphi_min() << " ";
    file << std::setprecision(precision) << MMentry.second.modules23().phi_slope_max() << " " << std::setprecision(precision) << MMentry.second.modules23().phi_slope_min() << " ";
    file << std::setprecision(precision) << MMentry.second.modules23().deta_max() << " " << std::setprecision(precision) << MMentry.second.modules23().deta_min() << " ";
    file << std::setprecision(precision) << MMentry.second.diff_dzdr_max() << " " << std::setprecision(precision) << MMentry.second.diff_dzdr_min () << " ";
    file << std::setprecision(precision) << MMentry.second.diff_dydx_max() << " " << std::setprecision(precision) << MMentry.second.diff_dydx_min ();
    file << std::endl;
  }
  file.close();

  file = std::ofstream ((MM_output+".doublets.txt").c_str(), std::ios::out);
  assert (!file.fail());
  for (std::pair <const std::vector<uint64_t>, module_doublet<T> >& MMentry : _module_map_doublet) {
    file << std::setprecision(precision) << MMentry.first[0] << " " << std::setprecision(precision) << MMentry.first[1] << " ";
    file << std::setprecision(precision) << MMentry.second.z0_max() << " " << std::setprecision(precision) << MMentry.second.z0_min() << " ";
    file << std::setprecision(precision) << MMentry.second.dphi_max() << " " << std::setprecision(precision) << MMentry.second.dphi_min() << " ";
    file << std::setprecision(precision) << MMentry.second.phi_slope_max() << " " << std::setprecision(precision) << MMentry.second.phi_slope_min() << " ";
    file << std::setprecision(precision) << MMentry.second.deta_max() << " " << std::setprecision(precision) << MMentry.second.deta_min() << " ";
    file << std::endl;
  }
  file.close();
}


//------------------------------------------------------------
template <class T>
void module_map_triplet<T>::save_TTree (std::string MM_output)
//------------------------------------------------------------
{
  clock_t start, end;
  start = clock();

  uint64_t module1, module2, module3;
  module1 = module2 = module3 = 0;
  module_triplet<T> ModuleTriplet;

  std::cout << "Save the module map as " << MM_output << std::endl;

  //  string path_RootFile = joinPaths(outputDir, MMname);
  TFile* RootFile = TFile::Open ((MM_output + ".triplets.root").c_str(), "RECREATE");
  TTree* TreeModuleTriplets = new TTree ("TreeModuleTriplet", "Tree containing the module triplet' features");

  //Set up the branches
  TreeModuleTriplets -> Branch ("Module1", &module1, "Module1/g");
  TreeModuleTriplets -> Branch ("Module2", &module2, "Module2/g");
  TreeModuleTriplets -> Branch ("Module3", &module3, "Module3/g");
  ModuleTriplet.branch (TreeModuleTriplets);

  //Save module per module connection in the MM (because of the 1GB limit per buffer in ROOT)
  int size = _module_map_triplet.size();
  for (std::pair <const std::vector<uint64_t>, module_triplet<T> >  &MMentry : _module_map_triplet)
    { module1 = MMentry.first[0]; //mmentry.first[0];
      module2 = MMentry.first[1];
      module3 = MMentry.first[2];

      std::vector<uint64_t> triplet = {module1, module2, module3};
      ModuleTriplet = MMentry.second;

      TreeModuleTriplets -> Fill();
    }

  RootFile -> cd ();
  TreeModuleTriplets -> Write ();
  RootFile -> Close (); // also deletes TTree if not already deleted

  // save Module Map Doublets
  module_doublet<T> ModuleDoublet;
  RootFile = TFile::Open ((MM_output + ".doublets.root").c_str(), "RECREATE");
  TTree* TreeModuleDoublets = new TTree ("TreeModuleDoublet", "Tree containing the module doublet' features");

  // Set up the branches
  TreeModuleDoublets -> Branch ("Module1", &module1, "Module1/g");
  TreeModuleDoublets -> Branch ("Module2", &module2, "Module2/g");
  ModuleDoublet.branch (TreeModuleDoublets, "12");

  size = _module_map_doublet.size();
  for (std::pair <const std::vector<uint64_t>, module_doublet<T> >  &MMentry : _module_map_doublet)
    { module1 = MMentry.first[0]; //mmentry.first[0];
      module2 = MMentry.first[1];

      std::vector<uint64_t> doublet = {module1, module2};
      ModuleDoublet = MMentry.second;

      TreeModuleDoublets -> Fill();
    }

  RootFile -> cd ();
  TreeModuleDoublets -> Write ();
  RootFile -> Close (); // also deletes TTree if not already deleted

  end = clock ();
  std::cout << "Module Map saved to disk" << std::endl;
  std::cout << "cpu time : " << (long double)(end-start)/CLOCKS_PER_SEC << std::endl;
}


//-----------------------------------------------------------
template <class T>
void module_map_triplet<T>::read_TTree (std::string MM_input)
//-----------------------------------------------------------
{
  clock_t start, end;
  start = clock();
  std::cout << "Reading the Module Map: " << MM_input << std::endl;

  uint64_t module1, module2, module3;
  module_triplet<T> ModuleTriplet;

  TFile* RootFile = TFile::Open ((MM_input + ".triplets.root").c_str(), "READ");
  if (!RootFile) throw std::invalid_argument ("Cannot open file " + MM_input + ".triplet.root");

  TTree* TreeModuleTriplets = (TTree*) RootFile -> Get ("TreeModuleTriplet");

  //Set up the branches
  TreeModuleTriplets -> SetBranchAddress ("Module1", &module1);
  TreeModuleTriplets -> SetBranchAddress ("Module2", &module2);
  TreeModuleTriplets -> SetBranchAddress ("Module3", &module3);
  ModuleTriplet.set_branch_address (TreeModuleTriplets);

  int nb_entries = TreeModuleTriplets -> GetEntries();
  std::cout << "module map triplet: nb entries = " << nb_entries << std::endl;

  for (int connection=0; connection < nb_entries; connection++)
    { TreeModuleTriplets -> GetEntry (connection);
      std::vector<uint64_t> triplet = {module1, module2, module3};
      _module_map_triplet.insert (std::pair<std::vector<uint64_t>, module_triplet<T>> (triplet, ModuleTriplet));
    }

  RootFile -> Close ("R");

  // reading the Module Map Doublet
  module_doublet<T> ModuleDoublet;
  RootFile = TFile::Open ((MM_input + ".doublets.root").c_str(), "READ");
  if (!RootFile) throw std::invalid_argument ("Cannot open file " + MM_input + "doublet.root");

  TTree* TreeModuleDoublets = (TTree*) RootFile -> Get ("TreeModuleDoublet");

  //Set up the branches
  TreeModuleDoublets -> SetBranchAddress ("Module1", &module1);
  TreeModuleDoublets -> SetBranchAddress ("Module2", &module2);
  ModuleDoublet.set_branch_address (TreeModuleDoublets, "12");

  nb_entries = TreeModuleDoublets -> GetEntries();
  std::cout << "module map doublet: nb entries = " << nb_entries << std::endl;

  for (int connection=0; connection < nb_entries; connection++)
    { TreeModuleDoublets -> GetEntry (connection);
      std::vector<uint64_t> doublet = {module1, module2};
      _module_map_doublet.insert (std::pair<std::vector<uint64_t>, module_doublet<T>> (doublet, ModuleDoublet));
      _module_map_pairs.insert (std::pair<std::vector<uint64_t>, int> (doublet, connection));
    }

  RootFile -> Close ("R");

  // map modules
  int loop = 0;
  for (std::pair<std::vector<uint64_t>, module_doublet<T>> ModuleDoublet : _module_map_doublet) {
    if (_module_map.find(ModuleDoublet.first[0]) == _module_map.end()) _module_map.insert (std::pair<uint64_t, int> (ModuleDoublet.first[0],0));
    if (_module_map.find(ModuleDoublet.first[1]) == _module_map.end()) _module_map.insert (std::pair<uint64_t, int> (ModuleDoublet.first[1],0));
    loop++;
  }

  loop=0;
  for (std::multimap<uint64_t,int>::iterator it = _module_map.begin(); it != _module_map.end(); ++it, loop++)
    it->second = loop;

  std::cout << red << "# modules in MM = " << _module_map.size() << reset;

  end = clock();
  std::cout << "Module Map loaded in memory" << std::endl;
  std::cout << "cpu time : " << (long double)(end-start)/CLOCKS_PER_SEC << std::endl;
}
