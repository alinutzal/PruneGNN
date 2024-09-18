/***************************************************************
 * Tracking project library - L2IT
 * Trace reconstruction in LHC
 * copyright © 2023 ROUGIER Charline
 * copyright © 2023,2024 COLLARD Christophe
 * copyright © 2023,2024 Centre National de la Recherche Scientifique
 * copyright © 2023,2024 Laboratoire des 2 Infinis de Toulouse (L2IT)
 ***************************************************************/

#ifndef __cplusplus
#error Must use C++ for the type geometric cuts
#endif

#include <iostream>
#include <TTree_hits>
#include <TMath.h>

//=====================================
template <class T> class geometric_cuts
//=====================================
{
  private:
    T _z0;
    T _dphi;
    T _phi_slope;
    T _r_phi_slope;
    T _deta;

  public:
    geometric_cuts () {}
    geometric_cuts (T, T, T, T);
    template <class V> geometric_cuts (const TTree_hits<V>&, const int&, const int&);
    ~geometric_cuts () {}

    void branch (TTree*, const std::string&, const std::string&);
    void set_branch_address (TTree*, const std::string&, const std::string&);

    template <class Tf> friend bool operator == (const geometric_cuts<Tf>&, const geometric_cuts<Tf>&);
    template <class Tf> friend bool operator != (const geometric_cuts<Tf>&, const geometric_cuts<Tf>&);
    template <class Tf> friend std::ostream& operator << (std::ostream&, const geometric_cuts<Tf>&);

    inline const T& z0 () const {return _z0;}
    inline const T& d_phi () const {return _dphi;}
    inline const T& phi_slope () const {return _phi_slope;}
    inline const T& r_phi_slope () const {return _r_phi_slope;}
    inline const T& d_eta () const {return _deta;}

    inline T& z0 () {return _z0;}
    inline T& d_phi () {return _dphi;}
    inline T& phi_slope () {return _phi_slope;}
    inline T& d_eta () {return _deta;}

    bool check_cut_values () const;
    void min (const geometric_cuts<T>&);
    void max (const geometric_cuts<T>&);
    inline bool min_cuts (const geometric_cuts<T>& cuts) {return _z0 > cuts._z0 || _dphi > cuts._dphi || _phi_slope > cuts._phi_slope || _deta > cuts._deta;}
    inline bool max_cuts (const geometric_cuts<T>& cuts) {return _z0 < cuts._z0 || _dphi < cuts._dphi || _phi_slope < cuts._phi_slope || _deta < cuts._deta;}
};
