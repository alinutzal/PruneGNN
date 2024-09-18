/***************************************************************
 * Tracking project library - L2IT
 * Trace reconstruction in LHC
 * copyright © 2022,2023 ROUGIER Charline
 * copyright © 2022,2023,2024 COLLARD Christophe
 * copyright © 2022,2023,2024 Centre National de la Recherche Scientifique
 * copyright © 2022,2023.2024 Laboratoire des 2 Infinis de Toulouse (L2IT)
 ***************************************************************/

#ifndef __cplusplus
#error Must use C++ for the type graph true
#endif

#include <colors>
#include <boost/graph/adjacency_list.hpp>
#include <boost/property_map/dynamic_property_map.hpp>
#include <boost/graph/graphml.hpp>
#include <TTree_hits>
#include <TTree_particles>
#include <vertex_true>
#include <edge_true>
#include <graph>

//    using graph_true = boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, VertexPropertyTrue,EdgePropertyTrue>;
//    typedef boost::graph_traits<graph_true>::edge_descriptor edge_true; 

template <class T> using graph_true_type = boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, vertex_true<T>, edge_true<T>>;
//template <class T> using vertex_true_type = typename boost::graph_traits<graph_true_type<T>>::vertex_descriptor;
template <class T> using edge_true_type = typename boost::graph_traits<graph_true_type<T>>::edge_descriptor;
template <class T> using edge_true_iterator = typename boost::graph_traits<graph_true_type<T>>::edge_iterator;

//=================================
template <class T> class graph_true
//=================================
{
  protected:
    graph_true_type<T> G;

  public:
    graph_true () {}
    ~graph_true () {}

    edge_true<T>& operator [] (const edge_true_type<T>& e) {return G[e];}
    vertex_true<T>& operator [] (const vertex_type<T>& v) {return G[v];}
//    void add_node (const hit<T>&);
    void add_node (const TTree_hits<T>&);
    void add_edge (const vertex_type<T>&, const vertex_type<T>&, int&, const T&, const uint64_t&, const int&, const int&, const std::pair<std::_Rb_tree_iterator<std::pair<const uint64_t, int>>, std::_Rb_tree_iterator<std::pair<const uint64_t, int>>>&, TTree_hits<T>&, TTree_particles<T>&);
    void modify_flag (vertex_type<T>&, vertex_type<T>&, int&, const T&, const uint64_t&, const int&, const int&, const std::pair<std::_Rb_tree_iterator<std::pair<const uint64_t, int>>, std::_Rb_tree_iterator<std::pair<const uint64_t, int>>>&, TTree_hits<T>& TThits, TTree_particles<T>& TTp);
    inline int num_edges () {return boost::num_edges(G);}
    inline int num_vertices () { return boost::num_vertices(G);}
    inline std::pair<edge_true_type<T>, bool> get_edge (vertex_type<T>& v1, vertex_type<T>& v2) {return boost::edge (v1, v2, G);}
    void save (const std::string&, const std::string&);
    void save_npz (const std::string&, const std::string&);
};
