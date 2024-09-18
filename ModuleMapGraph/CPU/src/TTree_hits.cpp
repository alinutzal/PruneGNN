/***************************************************************
 * Tracking project library - L2IT
 * Trace reconstruction in LHC
 * copyright © 2022,2023 ROUGIER Charline
 * copyright © 2022,2023,2024 COLLARD Christophe
 * copyright © 2022,2023,2024 Centre National de la Recherche Scientifique
 * copyright © 2022,2023,2024 Laboratoire des 2 Infinis de Toulouse (L2IT)
 ***************************************************************/

#include "TTree_hits.hpp"


//------------------------------------
template <class T>
void TTree_hits<T>::allocate_memory ()
//------------------------------------
{
  _module_ID = new std::vector <uint64_t>;
  _hardware = new std::vector<std::string>;
  _barrel_endcap = new std::vector<int>;
  _particle_ID1 = new std::vector<uint64_t>;
  _particle_ID2 = new std::vector<uint64_t>;

  if (_extra_features) {
    _layer_disk = new std::vector<int>;
    _eta_module = new std::vector<int>;
    _phi_module = new std::vector<int>;
    _cluster1_x = new std::vector<T>;
    _cluster1_y = new std::vector<T>;
    _cluster1_z = new std::vector<T>;
    _cluster2_x = new std::vector<T>;
    _cluster2_y = new std::vector<T>;
    _cluster2_z = new std::vector<T>;

    _R_cluster1 = new std::vector<T>;
    _eta_cluster1 = new std::vector<T>;
    _phi_cluster1 = new std::vector<T>;
    _R_cluster2 = new std::vector<T>;
    _eta_cluster2 = new std::vector<T>;
    _phi_cluster2 = new std::vector<T>;
  }

  _size = 0;
  _position = -1;
}


//----------------------------------------------------------
template <class T>
TTree_hits<T>::TTree_hits (bool true_graph, bool extra_feat)
//----------------------------------------------------------
{
  _true_graph = true_graph;
  _extra_features = extra_feat;
  allocate_memory();
}


//--------------------------------------------------
template <class T>
TTree_hits<T>::TTree_hits (const TTree_hits<T>& TTh)
//--------------------------------------------------
{
  _size = 0;
  _position = -1;
  *this = TTh;
}


/*
//--------------------------------------
template <class T>
TTree_hits<T>::TTree_hits (hits<T>& hts)
//--------------------------------------
{
int i = 0;
  allocate_memory();
  for (std::_Rb_tree_iterator <std::pair <const uint64_t, hit<T> > > it=hts.begin(); it!=hts.end(); ++it)
  { std::cout << "i = " <<  i << std::endl;
    push_back (it->second);
    i++;
  }
}
*/


//---------------------------
template <class T>
TTree_hits<T>::~TTree_hits ()
//---------------------------
{
  _size = 0;
  _position = -1;
  delete _module_ID;
  delete _hardware;
  delete _barrel_endcap;
  delete _particle_ID1;
  delete _particle_ID2;

  if (_extra_features) {
    delete _layer_disk;
    delete _eta_module;
    delete _phi_module;
    delete _cluster1_x;
    delete _cluster1_y;
    delete _cluster1_z;
    delete _cluster2_x;
    delete _cluster2_y;
    delete _cluster2_z;
    delete _R_cluster1;
    delete _eta_cluster1;
    delete _phi_cluster1;
    delete _R_cluster2;
    delete _eta_cluster2;
    delete _phi_cluster2;
  }
}


//-----------------------------------------------------------------
template <class T>
TTree_hits<T>& TTree_hits<T>::operator = (const TTree_hits<T>& TTh)
//-----------------------------------------------------------------
{
  assert (!_size || _size == TTh._size);
  assert (TTh._size);

  if (!_size) allocate_memory();
  _size = TTh._size;
  _position = TTh._position;
  _true_graph = TTh._true_graph;
  _extra_features = TTh._extra_features;

  _hit_id = TTh._hit_id;
  _x = TTh._x;
  _y = TTh._y;
  _z = TTh._z;
  _particle_ID = TTh._particle_ID;
  *_module_ID = *TTh._module_ID;
  *_hardware = *TTh._hardware;
  *_barrel_endcap = *TTh._barrel_endcap;
  *_particle_ID1 = *TTh._particle_ID1;
  *_particle_ID2 = *TTh._particle_ID2;

  _R = TTh._R;
  _eta = TTh._eta;
  _phi = TTh._phi;

  if (_extra_features) {
    *_layer_disk = *TTh._layer_disk;
    *_eta_module = *TTh._eta_module;
    *_phi_module = *TTh._phi_module;
    *_cluster1_x = *TTh._cluster1_x;
    *_cluster1_y = *TTh._cluster1_y;
    *_cluster1_z = *TTh._cluster1_z;
    *_cluster2_x = *TTh._cluster2_x;
    *_cluster2_y = *TTh._cluster2_y;
    *_cluster2_z = *TTh._cluster2_z;

    *_R_cluster1 = *TTh._R_cluster1;
    *_eta_cluster1 = *TTh._eta_cluster1;
    *_phi_cluster1 = *TTh._phi_cluster1;
    *_R_cluster2 = *TTh._R_cluster2;
    *_eta_cluster2 = *TTh._eta_cluster2;
    *_phi_cluster2 = *TTh._phi_cluster2;
  }

  _moduleID_TTreeHits_map = TTh._moduleID_TTreeHits_map;
  _particleID_TTreeHits_map = TTh._particleID_TTreeHits_map;

  return *this;
}


//-----------------------------------------------------------------------
template <class Tf>
bool operator == (const TTree_hits<Tf>& TTh1, const TTree_hits<Tf>& TTh2)
//-----------------------------------------------------------------------
{
  bool test = (TTh1._size >= 0);
  test *= (TTh1._size == TTh2._size);
  test *= (TTh1._extra_features == TTh2._extra_features);

  if (test) {
    test *= (TTh1._hit_id == TTh2._hit_id);
    test *= (TTh1._x == TTh2._x);
    test *= (TTh1._y == TTh2._y);
    test *= (TTh1._z == TTh2._z);
    test *= (TTh1._particle_ID == TTh2._particle_ID);
    test *= (*TTh1._module_ID == *TTh2._module_ID);
    test *= (*TTh1._hardware == *TTh2._hardware);
    test *= (*TTh1._barrel_endcap == *TTh2._barrel_endcap);
    test *= (*TTh1._particle_ID1 == *TTh2._particle_ID1);
    test *= (*TTh1._particle_ID2 == *TTh2._particle_ID2);

    test *= (TTh1._moduleID_TTreeHits_map == TTh2._moduleID_TTreeHits_map);
    test *= (TTh1._particleID_TTreeHits_map == TTh1._particleID_TTreeHits_map);

    if (TTh1._extra_features) {
      test *= (*TTh1._layer_disk == *TTh2._layer_disk);
      test *= (*TTh1._eta_module == *TTh2._eta_module);
      test *= (*TTh1._phi_module == *TTh2._phi_module);
      test *= (*TTh1._cluster1_x == *TTh2._cluster1_x);
      test *= (*TTh1._cluster1_y == *TTh2._cluster1_y);
      test *= (*TTh1._cluster1_z == *TTh2._cluster1_z);
      test *= (*TTh1._cluster2_x == *TTh2._cluster2_x);
      test *= (*TTh1._cluster2_y == *TTh2._cluster2_y);
      test *= (*TTh1._cluster2_z == *TTh2._cluster2_z);
    }
  }

  return test;
}


//-----------------------------------------------------------------------
template <class Tf>
bool operator != (const TTree_hits<Tf>& TTh1, const TTree_hits<Tf>& TTh2)
//-----------------------------------------------------------------------
{
  return !(TTh1==TTh2);
}


//-----------------------------------
template <class T>
void TTree_hits<T>::get (int i) const
//-----------------------------------
{
  assert (i>=0 && i<_size);
  _position = i;
}


//------------------------------
template <class T>
hit<T> TTree_hits<T>::get_hit ()
//------------------------------
{
  if (_extra_features)
    return hit<T> (_hit_id[_position], _x[_position], _y[_position], _z[_position], _particle_ID[_position], (*_module_ID)[_position], (*_hardware)[_position], (*_barrel_endcap)[_position], (*_particle_ID1)[_position], (*_particle_ID2)[_position],
      (*_layer_disk)[_position], (*_eta_module)[_position], (*_phi_module)[_position], (*_cluster1_x)[_position], (*_cluster1_y)[_position], (*_cluster1_z)[_position], (*_cluster2_x)[_position], (*_cluster2_y)[_position], (*_cluster2_z)[_position]);
  else
    return hit<T> (_hit_id[_position], _x[_position], _y[_position], _z[_position], _particle_ID[_position], (*_module_ID)[_position], (*_hardware)[_position], (*_barrel_endcap)[_position], (*_particle_ID1)[_position], (*_particle_ID2)[_position]);
}


//-------------------------------------------------
template <class T>
const uint64_t& TTree_hits<T>::hit_id (int i) const
//-------------------------------------------------
{
  assert (i>=0 && i<_size);

  return _hit_id[i];
}


//-------------------------------------
template <class T>
const T& TTree_hits<T>::x (int i) const
//-------------------------------------
{
  assert (i>=0 && i<_size);

  return _x[i];
}


//-------------------------------------
template <class T>
const T& TTree_hits<T>::y (int i) const
//-------------------------------------
{
  assert (i>=0 && i<_size);

  return _y[i];
}


//-------------------------------------
template <class T>
const T& TTree_hits<T>::z (int i) const
//-------------------------------------
{
  assert (i>=0 && i<_size);

  return _z[i];
}


//------------------------------------------------------
template <class T>
const uint64_t& TTree_hits<T>::particle_ID (int i) const
//------------------------------------------------------
{
  assert (i>=0 && i<_size);

  return _particle_ID[i];
}


//----------------------------------------------------
template <class T>
const uint64_t& TTree_hits<T>::module_ID (int i) const
//----------------------------------------------------
{
  assert (i>=0 && i<_size);

  return (*_module_ID)[i];
}


//------------------------------
template <class T>
T TTree_hits<T>::R (int i) const
//------------------------------
{
  assert (i>=0 && i<_size);

  return _R[i];
}


//--------------------------------
template <class T>
T TTree_hits<T>::Eta (int i) const
//--------------------------------
{
  assert (i>=0 && i<_size);

  return _eta[i];
}


//--------------------------------
template <class T>
T TTree_hits<T>::Phi (int i) const
//--------------------------------
{
  assert (i>=0 && i<_size);

  return _phi[i];
}


//------------------------------------------------------
template <class T>
const std::string& TTree_hits<T>::hardware (int i) const
//------------------------------------------------------
{
  assert (i>=0 && i<_size);

  return (*_hardware)[i];
}


//---------------------------------------------------
template <class T>
const int& TTree_hits<T>::barrel_endcap (int i) const
//---------------------------------------------------
{
  assert (i>=0 && i<_size);

  return (*_barrel_endcap)[i];
}


//------------------------------------------------
template <class T>
const int& TTree_hits<T>::layer_disk (int i) const
//------------------------------------------------
{
  assert (_extra_features);
  assert (i>=0 && i<_size);

  return (*_layer_disk)[i];
}


//------------------------------------------------
template <class T>
const int& TTree_hits<T>::eta_module (int i) const
//------------------------------------------------
{
  assert (_extra_features);
  assert (i>=0 && i<_size);

  return (*_eta_module)[i];
}


//------------------------------------------------
template <class T>
const int& TTree_hits<T>::phi_module (int i) const
//------------------------------------------------
{
  assert (_extra_features);
  assert (i>=0 && i<_size);

  return (*_phi_module)[i];
}


//----------------------------------------------
template <class T>
const T& TTree_hits<T>::x_cluster1 (int i) const
//----------------------------------------------
{
  assert (_extra_features);
  assert (i>=0 && i<_size);

  return (*_cluster1_x)[i];
}


//----------------------------------------------
template <class T>
const T& TTree_hits<T>::y_cluster1 (int i) const
//----------------------------------------------
{
  assert (_extra_features);
  assert (i>=0 && i<_size);

  return (*_cluster1_y)[i];
}


//----------------------------------------------
template <class T>
const T& TTree_hits<T>::z_cluster1 (int i) const
//----------------------------------------------
{
  assert (_extra_features);
  assert (i>=0 && i<_size);

  return (*_cluster1_z)[i];
}


//----------------------------------------------
template <class T>
const T& TTree_hits<T>::x_cluster2 (int i) const
//----------------------------------------------
{
  assert (_extra_features);
  assert (i>=0 && i<_size);

  return (*_cluster2_x)[i];
}


//----------------------------------------------
template <class T>
const T& TTree_hits<T>::y_cluster2 (int i) const
//----------------------------------------------
{
  assert (_extra_features);
  assert (i>=0 && i<_size);

  return (*_cluster2_y)[i];
}


//----------------------------------------------
template <class T>
const T& TTree_hits<T>::z_cluster2 (int i) const
//----------------------------------------------
{
  assert (_extra_features);
  assert (i>=0 && i<_size);

  return (*_cluster2_z)[i];
}


//---------------------------------------
template <class T>
T TTree_hits<T>::R_cluster1 (int i) const
//---------------------------------------
{
  assert (_extra_features);
  assert (i>=0 && i<_size);

  return (*_R_cluster1)[i];
}


//-----------------------------------------
template <class T>
T TTree_hits<T>::Eta_cluster1 (int i) const
//-----------------------------------------
{
  assert (_extra_features);
  assert (i>=0 && i<_size);

  return (*_eta_cluster1)[i];
}


//-----------------------------------------
template <class T>
T TTree_hits<T>::Phi_cluster1 (int i) const
//-----------------------------------------
{
  assert (_extra_features);
  assert (i>=0 && i<_size);

  return (*_phi_cluster1)[i];
}


//---------------------------------------
template <class T>
T TTree_hits<T>::R_cluster2 (int i) const
//---------------------------------------
{
  assert (_extra_features);
  assert (i>=0 && i<_size);

  return (*_R_cluster2)[i];
}


//-----------------------------------------
template <class T>
T TTree_hits<T>::Eta_cluster2 (int i) const
//-----------------------------------------
{
  assert (_extra_features);
  assert (i>=0 && i<_size);

  return (*_eta_cluster2)[i];
}


//-----------------------------------------
template <class T>
T TTree_hits<T>::Phi_cluster2 (int i) const
//-----------------------------------------
{
  assert (_extra_features);
  assert (i>=0 && i<_size);

  return (*_phi_cluster2)[i];
}


//-----------------------------------------------------------------------------------------
template <class Tf>
Tf Diff_dydx (const TTree_hits<Tf>& TThits, const int& it1, const int& it2, const int& it3)
//-----------------------------------------------------------------------------------------
{
//////////////////////////////////////////
// warning : recalculate the MM and then the nb of edges
// !!!!!!!!!! typo dy_12 = y1 - y2 !!!!!!
//////////////////////////////////////////
  Tf dy_12 = TThits._y[it2] - TThits._y[it1];
  Tf dy_23 = TThits._y[it2] - TThits._y[it3];
  Tf dx_12 = TThits._x[it1] - TThits._x[it2];
  Tf dx_23 = TThits._x[it2] - TThits._x[it3];

  Tf diff_dydx;
  if (dx_12 && dx_23)
    diff_dydx = (dy_12 / dx_12) - (dy_23 / dx_23);
  else if (dx_12)
    diff_dydx = ((dy_12 / dx_12) >= 0) ? std::numeric_limits<Tf>::max() : -std::numeric_limits<Tf>::max();
  else if (dx_23)
    diff_dydx = ((-dy_23 / dx_23) >= 0) ? std::numeric_limits<Tf>::max() : -std::numeric_limits<Tf>::max();
  else
    diff_dydx = 0;

    return diff_dydx;
}


//------------------------------------------------------------------------------------------
template <class Tf>
 Tf Diff_dzdr (const TTree_hits<Tf>& TThits, const int& it1, const int& it2, const int& it3)
//------------------------------------------------------------------------------------------
{
  Tf dz_12 = TThits._z[it2] - TThits._z[it1];
  Tf dz_23 = TThits._z[it3] - TThits._z[it2];
  Tf dr_12 = TThits._R[it2] - TThits._R[it1];
  Tf dr_23 = TThits._R[it3] - TThits._R[it2];

  Tf diff_dzdr;

  if (dr_12 && dr_23)
    diff_dzdr = (dz_12 / dr_12) - (dz_23 / dr_23);
  else if (dr_12)
    diff_dzdr = ((dz_12 / dr_12) >= 0 )? std::numeric_limits<Tf>::max() : -std::numeric_limits<Tf>::max();
  else if (dr_23)
    diff_dzdr = ((-dz_23 / dr_23) >= 0 )? std::numeric_limits<Tf>::max() : -std::numeric_limits<Tf>::max();
  else
    diff_dzdr = 0;

    return diff_dzdr;
}


//--//--------------------------------------------------------------------------------
template<class Tf>
int define_region (const TTree_hits<Tf>& TThits, const int& source, const int& target)
//------------------------------------------------------------------------------------
{
  std::string source_hardware = (*TThits._hardware)[source];
  std::string target_hardware = (*TThits._hardware)[target];
  int source_barrel_endcap = (*TThits._barrel_endcap)[source];
  int target_barrel_endcap = (*TThits._barrel_endcap)[target];

  // the two nodes are in the same region
  if ((source_hardware == target_hardware) && (source_barrel_endcap == target_barrel_endcap)) {
    //same region
    if (source_hardware == "PIXEL") {
      if (source_barrel_endcap == -2) return 1;
      if (source_barrel_endcap == 0) return 2;
      if (source_barrel_endcap == 2) return 3;
    }
    else if (source_hardware == "STRIP") {
      if (source_barrel_endcap == -2) return 4;
      if (source_barrel_endcap == 0) return 5;
      if (source_barrel_endcap == 2) return 6;
    }
  }
  // the two nodes are not in the same region but on the same sub-detector
  else
    if ((source_hardware == target_hardware)) {
      if (source_hardware == "PIXEL") {
        if ( (source_barrel_endcap == -2 && target_barrel_endcap == 0) || (target_barrel_endcap == -2 && source_barrel_endcap == 0) ) return 7;
        if ( (source_barrel_endcap == 2 && target_barrel_endcap == 0) || (target_barrel_endcap == 2 && source_barrel_endcap == 0) ) return 8;
        return 0;
      }
      else
        if (source_hardware == "STRIP") {
          if ((source_barrel_endcap == -2 && target_barrel_endcap == 0) || (target_barrel_endcap == -2 && source_barrel_endcap == 0)) return 9;
          if ( (source_barrel_endcap == 2 && target_barrel_endcap == 0) || (target_barrel_endcap == 2 && source_barrel_endcap == 0) ) return 10;
          return 0;
        }
      else return 0;
    }
  //not the same region neither the same sub-detector but the same barrel or endcaps
  else
    if (source_barrel_endcap == target_barrel_endcap) {
      if (source_barrel_endcap == 0) return 11;
      if (source_barrel_endcap == 2) return 12;
      if (source_barrel_endcap == -2) return 13;
      return 0;
    }
  else
    if ( (source_barrel_endcap ==0 && source_hardware == "PIXEL") || (target_barrel_endcap ==0 && target_hardware == "PIXEL")) {
      if ((source_barrel_endcap ==2 && source_hardware == "STRIP") || (target_barrel_endcap ==2 && target_hardware == "STRIP")) return 14;
      if ((source_barrel_endcap ==-2 && source_hardware == "STRIP") || (target_barrel_endcap ==-2 && target_hardware == "STRIP")) return 15;
    }
  else return 0;

  return 0;
}


//------------------------------------------
template <class T>
void TTree_hits<T>::branch (TTree* treeHits)
//------------------------------------------
{
  treeHits -> Branch ("hit_id", &_hit_id);
  treeHits -> Branch ("particle_ID", &_particle_ID);
//  std::string vector_Tname = "vector<" + boost::typeindex::type_id<T>().pretty_name() + ">";
//  treeHits -> Branch ("x", vector_Tname.c_str(), &_x);
  treeHits -> Branch ("x", &_x);
  treeHits -> Branch ("y", &_y);
  treeHits -> Branch ("z", &_z);
  treeHits -> Branch ("ID", &_module_ID);
  treeHits -> Branch ("hardware", &_hardware);
  treeHits -> Branch ("barrel_endcap", &_barrel_endcap);
  treeHits -> Branch ("particle_ID1", &_particle_ID1);
  treeHits -> Branch ("particle_ID2", &_particle_ID2);

  if (_extra_features) {
    treeHits -> Branch ("layer_disk", &_layer_disk);
    treeHits -> Branch ("eta_module", &_eta_module);
    treeHits -> Branch ("phi_module", &_phi_module);
    treeHits -> Branch ("cluster1_x", &_cluster1_x);
    treeHits -> Branch ("cluster1_y", &_cluster1_y);
    treeHits -> Branch ("cluster1_z", &_cluster1_z);
    treeHits -> Branch ("cluster2_x", &_cluster2_x);
    treeHits -> Branch ("cluster2_y", &_cluster2_y);
    treeHits -> Branch ("cluster2_z", &_cluster2_z);
  }
}


//------------------------------------------------------
template <class T>
void TTree_hits<T>::set_branch_address (TTree* treeHits)
//------------------------------------------------------
{
  treeHits -> SetBranchAddress ("hit_id", &_hit_id);
  treeHits -> SetBranchAddress ("x", &_x);
  treeHits -> SetBranchAddress ("y", &_y);
  treeHits -> SetBranchAddress ("z", &_z);
  treeHits -> SetBranchAddress ("particle_ID", &_particle_ID);
  treeHits -> SetBranchAddress ("ID", &_module_ID);
  treeHits -> SetBranchAddress ("hardware", &_hardware);
  treeHits -> SetBranchAddress ("barrel_endcap", &_barrel_endcap);
  treeHits -> SetBranchAddress ("particle_ID1", &_particle_ID1);
  treeHits -> SetBranchAddress ("particle_ID2", &_particle_ID2);

  if (_extra_features) {
    treeHits -> SetBranchAddress ("layer_disk", &_layer_disk);
    treeHits -> SetBranchAddress ("eta_module", &_eta_module);
    treeHits -> SetBranchAddress ("phi_module", &_phi_module);
    treeHits -> SetBranchAddress ("cluster1_x", &_cluster1_x);
    treeHits -> SetBranchAddress ("cluster1_y", &_cluster1_y);
    treeHits -> SetBranchAddress ("cluster1_z", &_cluster1_z);
    treeHits -> SetBranchAddress ("cluster2_x", &_cluster2_x);
    treeHits -> SetBranchAddress ("cluster2_y", &_cluster2_y);
    treeHits -> SetBranchAddress ("cluster2_z", &_cluster2_z);
  }

//_size = _hit_id.size();
//_position = 0;
}


//----------------------------------------------
template <class T>
void TTree_hits<T>::push_back (const hit<T>& ht)
//----------------------------------------------
{
  assert (_extra_features == ht.extra_features());
 /// std::vector<uint64_t>::iterator it = std::find((*_hit_id).begin(), (*_hit_id).end(), ht.hit_id());
//  if (it != (*_hit_id).end())
//    std::cout << red << "WARNING: duplicate hit id " << ht.hit_id() << " not stored" << reset;
//  else {
 /// if (it == (*_hit_id).end()) {
    _hit_id.push_back (ht.hit_id());
    _x.push_back (ht.x());
    _y.push_back (ht.y());
    _z.push_back (ht.z());
    _particle_ID.push_back (ht.particle_id());
    (*_module_ID).push_back (ht.module_ID());
    (*_hardware).push_back (ht.hardware());
    (*_barrel_endcap).push_back (ht.barrel_endcap());
    (*_particle_ID1).push_back (ht.particle_ID1());
    (*_particle_ID2).push_back (ht.particle_ID2());

    data_type2 x = ht.x();
    data_type2 y = ht.y();
    data_type2 z = ht.z();
    data_type2 r = std::sqrt (pow (x, 2) + pow (y, 2));
    _R.push_back (r);

    data_type2 r3 = std::sqrt (pow(r, 2) + pow(z,2));
    data_type2 theta = 0.5 * acos (z / r3);
    data_type2 eta = -log (tan (theta));
    _eta.push_back (eta);

    data_type2 phi = atan2 (y, x);
    _phi.push_back (phi);

    if (_extra_features) {
      (*_layer_disk).push_back (ht.layer_disk());
      (*_eta_module).push_back (ht.eta_module());
      (*_phi_module).push_back (ht.phi_module());
      (*_cluster1_x).push_back (ht.cluster1_x());
      (*_cluster1_y).push_back (ht.cluster1_y());
      (*_cluster1_z).push_back (ht.cluster1_z());
      (*_cluster2_x).push_back (ht.cluster2_x());
      (*_cluster2_y).push_back (ht.cluster2_y());
      (*_cluster2_z).push_back (ht.cluster2_z());
      T r_cluster1 = std::sqrt (ht.cluster1_x() * ht.cluster1_x() + ht.cluster1_y() * ht.cluster1_y());
      (*_R_cluster1).push_back (r_cluster1);
      T r3_cluster1 = std::sqrt (r_cluster1 * r_cluster1 + ht.cluster1_z() * ht.cluster1_z()); // use U for template
      T theta_cluster1 = acos (ht.cluster1_z() / r3_cluster1);
      T eta_cluster1 = -log (tan (theta_cluster1 * 0.5));
      (*_eta_cluster1).push_back (eta_cluster1);
      (*_phi_cluster1).push_back (atan2 (ht.cluster1_y(), ht.cluster1_x()));
      T r_cluster2 = std::sqrt (ht.cluster2_x() * ht.cluster2_x() + ht.cluster2_y() * ht.cluster2_y());
      (*_R_cluster2).push_back (r_cluster2);
      T r3_cluster2 = std::sqrt (r_cluster2 * r_cluster2 + ht.cluster2_z() * ht.cluster2_z()); // use U for template
      T theta_cluster2 = acos (ht.cluster2_z() / r3_cluster2);
      T eta_cluster2 = -log (tan (theta_cluster2 * 0.5));
      (*_eta_cluster2).push_back (eta_cluster2);
      (*_phi_cluster2).push_back (atan2 (ht.cluster2_y(), ht.cluster2_x()));
    }

    _moduleID_TTreeHits_map.insert (std::pair<uint64_t, int> (ht.module_ID(), _size));
    _particleID_TTreeHits_map.insert (std::pair<uint64_t, int> (ht.particle_id(), _size));
    _size++;
///  }
}


//----------------------------------------------------
template <class T>
void TTree_hits<T>::save (const std::string& filename)
//----------------------------------------------------
{
  TFile* RootFile = TFile::Open (filename.c_str(), "RECREATE");
  TTree* treeEventHits = new TTree ("TreeEventHits", "Tree containing the hits of an event' features");

  // Set up the branches
  branch (treeEventHits);
  treeEventHits -> Fill();
  RootFile -> cd ();
  treeEventHits -> Write ();
  RootFile -> Close (); // also deletes TTree if not already deleted
  std::cout << "hits root file written in " << filename << std::endl;
}


//----------------------------------------------------
template <class T>
void TTree_hits<T>::read (const std::string& filename)
//----------------------------------------------------
{
  TFile* RootFile = TFile::Open (filename.c_str(), "READ");
  if (!RootFile) throw std::invalid_argument ("Cannot open file " + filename);

  TTree* treeEventHits = (TTree*) RootFile -> Get ("TreeEventHits");
  set_branch_address (treeEventHits);
//  std::cout << "nb entries = " << treeEventHits->GetEntries() << std::endl;
  _size = treeEventHits->GetEntries();

//    std::vector<uint64_t>* athena_moduleId;
//    vector<int>* index;
//    vector<int long>* barcode;
//    vector<int>* subevent;

//    std::vector<T> *_R, *_eta, *_phi;
//    std::vector<T> *_R_cluster1, *_eta_cluster1, *_phi_cluster1;
//    std::vector<T> *_R_cluster2, *_eta_cluster2, *_phi_cluster2;


  treeEventHits -> GetEntry(0);

  RootFile -> Close ();

  for (int i=0; i<_size; i++) {
    data_type2 xi = _x[i];
    data_type2 yi = _y[i];
    data_type2 r = sqrt (xi * xi + yi * yi);
    _R.push_back (r);

    data_type2 zi = _z[i];
    data_type2 r3 = std::sqrt (r * r + zi * zi);
    data_type2 theta = 0.5 * acos (zi / r3);
    data_type2 eta = -log (tan (theta));
    _eta.push_back (eta);
    data_type2 phi = atan2 (yi, xi);
    _phi.push_back (phi);

    if (_extra_features) {
      T xi_cluster1 = (*_cluster1_x)[i];
      T yi_cluster1 = (*_cluster1_y)[i];
      T r_cluster1 = sqrt (xi_cluster1 * xi_cluster1 + yi_cluster1 * yi_cluster1);
      (*_R_cluster1).push_back (r_cluster1);

      T zi_cluster1 = (*_cluster1_z)[i];
      T r3_cluster1 = std::sqrt (r_cluster1 * r_cluster1 + zi_cluster1 * zi_cluster1); // use U for template
      T theta_cluster1 = acos (zi_cluster1 / r3_cluster1);
      T eta_cluster1 = -log (tan (theta_cluster1 * 0.5));
      (*_eta_cluster1).push_back (eta_cluster1);
      (*_phi_cluster1).push_back (atan2 (yi_cluster1, xi_cluster1));

      T xi_cluster2 = (*_cluster2_x)[i];
      T yi_cluster2 = (*_cluster2_y)[i];
      T r_cluster2 = sqrt (xi_cluster2 * xi_cluster2 + yi_cluster2 * yi_cluster2);
      (*_R_cluster2).push_back (r_cluster2);

      T zi_cluster2 = (*_cluster2_z)[i];
      T r3_cluster2 = std::sqrt (r_cluster2 * r_cluster2 + zi_cluster2 * zi_cluster2); // use U for template
      T theta_cluster2 = acos (zi_cluster2 / r3_cluster2);
      T eta_cluster2 = -log (tan (theta_cluster2 * 0.5));
      (*_eta_cluster2).push_back (eta_cluster2);
      (*_phi_cluster2).push_back (atan2 (yi_cluster2, xi_cluster2));
    }

    if (_extra_features) {
      T xi_cluster1 = (*_cluster1_x)[i];
      T yi_cluster1 = (*_cluster1_y)[i];
      T r_cluster1 = sqrt (xi_cluster1 * xi_cluster1 + yi_cluster1 * yi_cluster1);
      (*_R_cluster1).push_back (r_cluster1);

      T zi_cluster1 = (*_cluster1_z)[i];
      T r3_cluster1 = std::sqrt (r_cluster1 * r_cluster1 + zi_cluster1 * zi_cluster1); // use U for template
      T theta_cluster1 = acos (zi_cluster1 / r3_cluster1);
      T eta_cluster1 = -log (tan (theta_cluster1 * 0.5));
      (*_eta_cluster1).push_back (eta_cluster1);
      (*_phi_cluster1).push_back (atan2 (yi_cluster1, xi_cluster1));

      T xi_cluster2 = (*_cluster2_x)[i];
      T yi_cluster2 = (*_cluster2_y)[i];
      T r_cluster2 = sqrt (xi_cluster2 * xi_cluster2 + yi_cluster2 * yi_cluster2);
      (*_R_cluster2).push_back (r_cluster2);

      T zi_cluster2 = (*_cluster2_z)[i];
      T r3_cluster2 = std::sqrt (r_cluster2 * r_cluster2 + zi_cluster2 * zi_cluster2); // use U for template
      T theta_cluster2 = acos (zi_cluster2 / r3_cluster2);
      T eta_cluster2 = -log (tan (theta_cluster2 * 0.5));
      (*_eta_cluster2).push_back (eta_cluster2);
      (*_phi_cluster2).push_back (atan2 (yi_cluster2, xi_cluster2));
    }

    _moduleID_TTreeHits_map.insert (std::pair<uint64_t, int> ((*_module_ID)[i], i));
    _particleID_TTreeHits_map.insert (std::pair<uint64_t, int> (_particle_ID[i], i));
  }

  //create_sorted_map ();
}
