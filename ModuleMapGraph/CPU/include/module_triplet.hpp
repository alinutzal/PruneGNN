/***************************************************************
 * Tracking project library - L2IT
 * Trace reconstruction in LHC
 * copyright © 2022,2023 ROUGIER Charline
 * copyright © 2022,2023,2024 COLLARD Christophe
 * copyright © 2022,2023,2024 Centre National de la Recherche Scientifique
 * copyright © 2022,2023,2024 Laboratoire des 2 Infinis de Toulouse (L2IT)
 ***************************************************************/

#ifndef __cplusplus
#error Must use C++ for the type module triplet
#endif

#include <iostream>
#include <TTree.h>
#include <geometric_cuts>
#include <module_doublet>

//=====================================
template <class T> class module_triplet
//=====================================
{
  private:
    unsigned int _occurence;

    module_doublet<T> _MD12, _MD23;

    T _diff_dydx_min;
    T _diff_dydx_max;
    T _diff_dzdr_min;
    T _diff_dzdr_max;

  public:
    module_triplet ();
    module_triplet (const geometric_cuts<T>&, const geometric_cuts<T>&, const T&, const T&);
//    module_triplet (const TTree_hits<T>&, const TTree_particles<T>&);
    ~module_triplet () {}

    template <class Tf> friend bool operator == (const module_triplet<Tf>&, const module_triplet<Tf>&);
    template <class Tf> friend bool operator != (const module_triplet<Tf>&, const module_triplet<Tf>&);
    // overload for iostream
    template <class Tf> friend std::ostream& operator << (std::ostream&, const module_triplet<Tf>&);
//    friend istream& operator >> (istream&, const module_triplet&);

    void branch (TTree*);
    void set_branch_address (TTree*);

    const inline unsigned int& occurence () const {return _occurence;}
    inline module_doublet<T>& modules12 () {return _MD12;}
    inline module_doublet<T>& modules23 () {return _MD23;}
    inline const module_doublet<T>& modules12 () const {return _MD12;}
    inline const module_doublet<T>& modules23 () const {return _MD23;}

    inline unsigned int& occurence () {return _occurence;}
    const T& diff_dydx_min () const {return _diff_dydx_min;}
    const T& diff_dydx_max () const {return _diff_dydx_max;}
    const T& diff_dzdr_min () const {return _diff_dzdr_min;}
    const T& diff_dzdr_max () const {return _diff_dzdr_max;}

    T& diff_dydx_min () {return _diff_dydx_min;}
    T& diff_dydx_max () {return _diff_dydx_max;}
    T& diff_dzdr_min () {return _diff_dzdr_min;}
    T& diff_dzdr_max () {return _diff_dzdr_max;}

    bool min_equal_max ();
    bool add_occurence (const geometric_cuts<T>&, const geometric_cuts<T>&, const T&, const T&);
    void add_occurence (const module_triplet<T>&);
    inline bool cuts_dydx (const T& diff_dydx)  {return diff_dydx < _diff_dydx_min  ||  diff_dydx > _diff_dydx_max;}
    inline bool cuts_dzdr (const T& diff_dzdr)  {return diff_dzdr < _diff_dzdr_min  ||  diff_dzdr > _diff_dzdr_max;}
};
