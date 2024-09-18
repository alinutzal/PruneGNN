/***************************************************************
 * Tracking project library - L2IT
 * Trace reconstruction in LHC
 * copyright © 2022,2023 ROUGIER Charline
 * copyright © 2022,2023,2024 COLLARD Christophe
 * copyright © 2022,2023,2024 Centre National de la Recherche Scientifique
 * copyright © 2022,2023,2024 Laboratoire des 2 Infinis de Toulouse (L2IT)
 ***************************************************************/

#ifndef __cplusplus
#error Must use C++ for the type module duet
#endif

#include <iostream>
#include <iomanip>
#include <TTree.h>
#include <particles> // "particles.hpp"
#include <geometric_cuts>

//=====================================
template <class T> class module_doublet
//=====================================
{
  private:
    geometric_cuts<T> _cuts_min;
    geometric_cuts<T> _cuts_max;

  public:
    module_doublet ();
    module_doublet (const geometric_cuts<T>&);
    ~module_doublet () {}

    template <class Tf> friend bool operator == (const module_doublet<Tf>&, const module_doublet<Tf>&);
    template <class Tf> friend bool operator != (const module_doublet<Tf>&, const module_doublet<Tf>&);
    // overload for iostream
    template <class Tf> friend std::ostream& operator << (std::ostream&, const module_doublet<Tf>&);
//    friend istream& operator >> (istream&, const module_triplet&);

    // overload for iostream
    template <class Tf> friend std::ostream& operator << (std::ostream&, const module_doublet<Tf>&);

    void branch (TTree*, const std::string&);
    void set_branch_address (TTree*, const std::string&);

    inline const T& z0_min () const {return _cuts_min.z0();}
    inline const T& z0_max () const {return _cuts_max.z0();}
    inline const T& dphi_min () const {return _cuts_min.d_phi();}
    inline const T& dphi_max () const {return _cuts_max.d_phi();}
    inline const T& phi_slope_min () const {return _cuts_min.phi_slope();}
    inline const T& phi_slope_max () const {return _cuts_max.phi_slope();}
    inline const T& deta_min () const {return _cuts_min.d_eta();}
    inline const T& deta_max () const {return _cuts_max.d_eta();}
    inline geometric_cuts<T> cuts_min () const {return _cuts_min;}
    inline geometric_cuts<T> cuts_max () const {return _cuts_max;}

    inline T& z0_min () {return _cuts_min.z0();}
    inline T& z0_max () {return _cuts_max.z0();}
    inline T& dphi_min () {return _cuts_min.d_phi();}
    inline T& dphi_max () {return _cuts_max.d_phi();}
    inline T& phi_slope_min () {return _cuts_min.phi_slope();}
    inline T& phi_slope_max () {return _cuts_max.phi_slope();}
    inline T& deta_min () {return _cuts_min.d_eta();}
    inline T& deta_max () {return _cuts_max.d_eta();}

    inline bool min_equal_max () {return _cuts_min == _cuts_max;}
    inline bool cuts (const geometric_cuts<T> SP_cuts) {return _cuts_min.min_cuts (SP_cuts) || _cuts_max.max_cuts (SP_cuts);}
    bool add_occurence (const geometric_cuts<T>&);
    void add_occurence (const module_doublet<T>&);
};
