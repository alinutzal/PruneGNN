/***************************************************************
 * Tracking project library - L2IT
 * Trace reconstruction in LHC
 * copyright © 2022,2023 ROUGIER Charline
 * copyright © 2022,2023,2024 COLLARD Christophe
 * copyright © 2022,2023,2024 Centre National de la Recherche Scientifique
 * copyright © 2022,2023,2024 Laboratoire des 2 Infinis de Toulouse (L2IT)
 ***************************************************************/

#ifndef __cplusplus
#error Must use C++ for the type module map triplet
#endif

#include <time.h>
#include <string>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <map>
#include <TFile.h>
#include <TTree.h>
#include <boost/program_options.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/regex.hpp>
#include <hits>
#include <TTree_hits>
#include <TTree_particles>
#include <hits_infos>
#include <geometric_cuts>
#include <module_doublet>
#include <module_triplet>
#include <edge>
#include <iterator>

//==========================================================
template <class T> class module_map_triplet : public TObject
//==========================================================
{
  private:
    bool _strip_hit_pair;
    // event nb / modules triplet nb (ex: 1,2,3) / module_triplet
    // std::multimap <std::pair<int, std::vector<uint64_t>, module_triplet<T,U>> _module_map_triplet;
    TTree_particles<T> _ModMap_particles;
    std::vector<geometric_cuts<T>> _cuts12, _cuts23;

    std::multimap <std::vector<uint64_t>, module_triplet<T>> _module_map_triplet;
    std::multimap <std::vector<uint64_t>, module_doublet<T>> _module_map_doublet;
    std::multimap <std::vector<uint64_t>, int> _module_map_pairs;
    std::multimap <uint64_t, int> _module_map;

//    std::multimap <std::vector<uint64_t>, unsigned int> _module_map_doublet;
//    module_doublet<T> *_module_doublets;
    ///    std::multimap <std::vector<uint64_t>, std::vector<int>> _moduleID_particles;
    std::multimap <std::vector<uint64_t>, std::vector<int>> _modules_particles; // module1, module2, module3 <-> particles position in TTp
    std::multimap <std::vector<uint64_t>, std::vector<T>> _modules_particlesKey; // probasbly useless ?
    std::multimap <std::vector<T>, int> _particlesSPECS_TTp_link;

    void add_event_triplets (const std::string&, const std::string&, const std::string&, const int&, const T&, const T&);

  public:
    module_map_triplet () {} //p = new _Rb_tree_iterator <pair <const vector<long long unsigned int>, module_triplet<float> > >;}
    module_map_triplet (boost::program_options::variables_map&, const std::string&, const std::string&);
    ~module_map_triplet () {} //delete p;}

    bool operator ! () {return !_module_map_triplet.size();}
//    bool operator += (const module_triplet<T>);
    template <class Tf> friend bool operator == (const module_map_triplet<Tf>&, const module_map_triplet<Tf>&);
    template <class Tf> friend bool operator != (const module_map_triplet<Tf>&, const module_map_triplet<Tf>&);
    inline std::_Rb_tree_iterator <std::pair <const std::vector<uint64_t>, module_triplet<T>>> begin() {return _module_map_triplet.begin();}
    inline std::_Rb_tree_iterator <std::pair <const std::vector<uint64_t>, module_triplet<T>>> end() {return _module_map_triplet.end();}
    inline int size () const {return _module_map_triplet.size();}

    inline const std::multimap <std::vector <uint64_t>, module_triplet<T>>* operator () () const {return &_module_map_triplet;}
    inline const std::multimap <std::vector<uint64_t> , module_triplet<T> >& map_triplet () const {return _module_map_triplet;}
    inline std::multimap <std::vector<uint64_t> , module_triplet<T> >& map_triplet () {return _module_map_triplet;}
    inline const std::multimap <std::vector<uint64_t> , module_doublet<T> >& map_doublet () const {return _module_map_doublet;}
    inline const std::multimap <std::vector<uint64_t>, int>& map_pairs () const {return _module_map_pairs;}
    inline int map_pair (const std::vector<uint64_t>& modules) {return _module_map_pairs.find (modules)->second;}
    inline std::multimap <uint64_t, int> module_map () {return _module_map;}
    inline std::multimap <std::vector<uint64_t> , module_doublet<T> >& map_doublet () {return _module_map_doublet;}
    inline std::vector<T> particles_key (const std::vector<uint64_t>& modules) {return _modules_particlesKey.find(modules);} // probably useless
    inline std::vector<int>& particles_position (const std::vector<uint64_t>& modules) {return _modules_particles.find(modules)->second;}
//    bool add_triplet_occurence (const std::vector<uint64_t>&, const TTree_hits<T>&, const TTree_particles<T>&, const geometric_cuts<T>&, const geometric_cuts<T>&, const T&, const T&);
    bool add_triplet_occurence (const std::vector<uint64_t>&, const TTree_hits<T>&, const TTree_particles<T>&, const geometric_cuts<T>&, const geometric_cuts<T>&, const T&, const T&, int, int, int);
    void merge (const module_map_triplet<T>&);
    void sort ();
    void remove_unique_occurence ();
    void remove_cycles ();

    void save_txt (std::string, const int=16);
    void save_TTree (std::string);
    void read_TTree (std::string);
};
