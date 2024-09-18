/***************************************************************
 * Tracking project library - L2IT
 * Trace reconstruction in LHC
 * copyright © 2022,2023 ROUGIER Charline
 * copyright © 2022,2023 COLLARD Christophe
 * copyright © 2022,2023 Centre National de la Recherche Scientifique
 * copyright © 2022,2023 Laboratoire des 2 Infinis de Toulouse (L2IT)
 ***************************************************************/

#ifndef __cplusplus
#error Must use C++ for the type vertex
#endif

#include <string>

template <class T> class graph;

//=============================
template <class T> class vertex
//=============================
{
  friend void graph<T>::save (const std::string&, const std::string&, const bool&);

  protected:
    T _r;
    T _phi;
    T _z;
    T _eta; 
    T _r_cluster1;
    T _phi_cluster1;
    T _z_cluster1;
    T _eta_cluster1; 
    T _r_cluster2;
    T _phi_cluster2;
    T _z_cluster2;
    T _eta_cluster2; 
    int _hit_id;

  public:
    vertex () {_r = _phi = _z = _eta = _hit_id = _r_cluster1 = _phi_cluster1 = _z_cluster1 = _eta_cluster1 = _r_cluster2 = _phi_cluster2 = _z_cluster2 = _eta_cluster2 = _hit_id = 0;}
    vertex (const T&, const T&, const T&, const T&);
    vertex (const T&, const T&, const T&, const T&, const T&, const T&, const T&, const T&, const T&, const T&, const T&, const T&, const T&);

    T& r () {return _r;}
    T& phi () {return _phi;}
    T& z () {return _z;}
    T& eta () {return _eta;}
    T& r_cluster1 () {return _r_cluster1;}
    T& phi_cluster1 () {return _phi_cluster1;}
    T& z_cluster1 () {return _z_cluster1;}
    T& eta_cluster1 () {return _eta_cluster1;}
    T& r_cluster2 () {return _r_cluster2;}
    T& phi_cluster2 () {return _phi_cluster2;}
    T& z_cluster2 () {return _z_cluster2;}
    T& eta_cluster2 () {return _eta_cluster2;}
    int& hit_id () {return _hit_id;}
};
