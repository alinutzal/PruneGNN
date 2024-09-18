/***************************************************************
 * Tracking project library - L2IT
 * Trace reconstruction in LHC
 * copyright © 2022,2023 ROUGIER Charline
 * copyright © 2022,2023,2024 COLLARD Christophe
 * copyright © 2022,2023,2024 Centre National de la Recherche Scientifique
 * copyright © 2022,2023,2024 Laboratoire des 2 Infinis de Toulouse (L2IT)
 ***************************************************************/

#ifndef __cplusplus
#error Must use C++ for the type graph
#endif

#include <boost/graph/adjacency_list.hpp>
#include <boost/property_map/dynamic_property_map.hpp>
#include <boost/graph/graphml.hpp>
#include <boost/python/numpy.hpp>
#include <vertex>
#include <edge>
#include <Python.h>
#include <torch/torch.h>
//#include <torch/script.h>

template <class T> using graph_type = boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, vertex<T>, edge<T>>;
template <class T> using vertex_type = typename boost::graph_traits<graph_type<T>>::vertex_descriptor;
template <class T> using vertex_iterator = typename boost::graph_traits<graph_type<T>>::vertex_iterator;
template <class T> using edge_type = typename boost::graph_traits<graph_type<T>>::edge_descriptor;
template <class T> using edge_iterator = typename boost::graph_traits<graph_type<T>>::edge_iterator;

//============================
template <class T> class graph
//============================
{
  protected:
    graph_type<T> G;

  public:
    graph () {}
    ~graph () {}

    void create (const TTree_hits<T>&, int, std::vector<int>*, std::vector<int>*, std::vector<int>&, std::vector<bool>*);
    vertex_type<T> add_node (const TTree_hits<T>&);
    vertex_type<T> add_vertex (const uint64_t&, const T&, const T&, const T&, const T&);
//    void add_node_graph_true (vertex_type<T>, const hit<T>&);
    void add_edge (const vertex_type<T>&, const vertex_type<T>&, const edge<T>&);
    void add_edge (const int&, const int&, const edge<T>&);
    inline int num_edges () {return boost::num_edges(G);}
    inline int num_vertices () { return boost::num_vertices(G);}
    inline std::pair<edge_type<T>, bool> get_edge (vertex_type<T>& v1, vertex_type<T>& v2) {return boost::edge (v1, v2, G);}
    void save (const std::string&, const std::string&, const bool&);
    void save_npz (const std::string&, const std::string&, const bool&);
    void save_pyg (const std::string&, const std::string&, const bool&);
    void save_csv (const std::string&, const std::string&, const bool&);
};
