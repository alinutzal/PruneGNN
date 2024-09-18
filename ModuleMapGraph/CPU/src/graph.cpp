/***************************************************************
 * Tracking project library - L2IT
 * Trace reconstruction in LHC
 * copyright © 2022,2023 ROUGIER Charline
 * copyright © 2022,2023,2024 COLLARD Christophe
 * copyright © 2022,2023,2024 Centre National de la Recherche Scientifique
 * copyright © 2022,2023,2024 Laboratoire des 2 Infinis de Toulouse (L2IT)
 ***************************************************************/

#include "graph.hpp"

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <class T>
void graph<T>::create (const TTree_hits<T>& TThits, int MMD_size, std::vector<int>* M1_SP, std::vector<int>* M2_SP, std::vector<int>& hit2vertex, std::vector<bool>* edges)
//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
{
  int nb_vertices = 0;
  vertex_type<T> vertices [hit2vertex.size()];
  for (int i=0; i<hit2vertex.size(); i++)
    if (hit2vertex[i]) {
      vertices[i] = boost::add_vertex (G);
      TThits.get (i);
      G[vertices[i]] = vertex<T> (TThits.R() * 0.001, TThits.Phi () / TMath::Pi(), TThits.z() * 0.001, TThits.hit_id());
      hit2vertex[i] = nb_vertices;
      nb_vertices++;
    }

  for (int i=0; i<MMD_size; i++)
    for (int j=0; j<edges[i].size(); j++)
      if (edges[i][j]) {
        edge_type<T> e;
        bool b;
        boost::tie (e, b) = boost::add_edge (vertices[M1_SP[i][j]], vertices[M2_SP[i][j]], G);
      }
}

//-------------------------------------------------------------
template <class T>
vertex_type<T> graph<T>::add_node (const TTree_hits<T>& TThits)
//-------------------------------------------------------------
{
  vertex_type<T> v = boost::add_vertex (G);
  if (TThits.extra_features())
    G[v] = vertex<T> (TThits.R() * 0.001, TThits.Phi () / TMath::Pi(), TThits.z() * 0.001, TThits.Eta(), TThits.hit_id(), TThits.R_cluster1() * 0.001, TThits.Phi_cluster1 () / TMath::Pi(), TThits.z_cluster1() * 0.001, TThits.Eta_cluster1(), TThits.R_cluster2() * 0.001, TThits.Phi_cluster2 () / TMath::Pi(), TThits.z_cluster2() * 0.001, TThits.Eta_cluster2());
  else
    G[v] = vertex<T> (TThits.R() * 0.001, TThits.Phi () / TMath::Pi(), TThits.z() * 0.001, TThits.hit_id());

  return v;
}

//--------------------------------------------------------------------------------------------------------------
template <class T>
vertex_type<T> graph<T>::add_vertex (const uint64_t& hit_id, const T& R, const T& z, const T& eta, const T& phi)
//--------------------------------------------------------------------------------------------------------------
{
  vertex_type<T> v = boost::add_vertex (G);
  G[v] = vertex<T> (R * 0.001, phi / TMath::Pi(), z * 0.001, hit_id);

  return v;
}

//----------------------------------------------------------------------------------------------------------------------------------
template <class T>
void graph<T>::add_edge (const vertex_type<T>& v1, const vertex_type<T>& v2, const edge<T>& edg) //const T& deta, const T& dphi, const T& dr, const T& dz)
//void graph<T>::add_edge (const vertex_type<T>& v1, const vertex_type<T>& v2, const edge_type<T>& edge) ????
//----------------------------------------------------------------------------------------------------------------------------------
{
  /*
    Add an edge and its 4 features (deta, dephi, dr, dz)
  */
  edge_type<T> e;
  bool b;
  boost::tie (e, b) = boost::add_edge (v1, v2, G);

//  edge<T> edg (deta, dphi, dr, dz);
  G[e] = edg;
//    G[e].dEta() = deta;
//    G[e].dPhi() = dphi;
//    G[e].dr() = dr;
//    G[e].dz() = dz;
}

//--------------------------------------------------------------------------------
template <class T>
void graph<T>::add_edge (const int& v1_loc, const int& v2_loc, const edge<T>& edg)
//--------------------------------------------------------------------------------
{
  vertex_iterator<T> vi, vi_end;
  edge_type<T> e;
  bool b;

  boost::tie (vi, vi_end) = boost::vertices (G);
  boost::tie (e, b) = boost::add_edge (vi[v1_loc], vi[v2_loc], G);
  G[e] = edg;
}

//----------------------------------------------------------------------------------------------------------
template <class T>
void graph<T>::save (const std::string& event_id, const std::string& output_dir, const bool& extra_features)
//----------------------------------------------------------------------------------------------------------
{
  boost::dynamic_properties dp (boost::ignore_other_properties);

  //edge
  dp.property ("dEta", boost::get (&edge<T>::_dEta, G));
  dp.property ("dPhi", boost::get (&edge<T>::_dPhi, G));
  dp.property ("dr", boost::get (&edge<T>::_dr, G));
  dp.property ("dz", boost::get (&edge<T>::_dz, G));
  dp.property ("phi_slope", boost::get (&edge<T>::_phi_slope, G));
  dp.property ("r_phi_slope", boost::get (&edge<T>::_r_phi_slope, G));
  if (extra_features) {
    dp.property ("new_hit_in_position_x", boost::get (&edge<T>::_new_hit_in_x, G));
    dp.property ("new_hit_in_position_y", boost::get (&edge<T>::_new_hit_in_y, G));
    dp.property ("new_hit_in_position_z", boost::get (&edge<T>::_new_hit_in_z, G));
    dp.property ("new_hit_out_position_x", boost::get (&edge<T>::_new_hit_out_x, G));
    dp.property ("new_hit_out_position_y", boost::get (&edge<T>::_new_hit_out_y, G));
    dp.property ("new_hit_out_position_z", boost::get (&edge<T>::_new_hit_out_z, G));
    dp.property ("pass_strip_range", boost::get (&edge<T>::_pass_strip_range, G));
    dp.property ("new_dEta", boost::get (&edge<T>::_new_dEta, G));
    dp.property ("new_dPhi", boost::get (&edge<T>::_new_dPhi, G));
    dp.property ("new_dr", boost::get (&edge<T>::_new_dr, G));
    dp.property ("new_dz", boost::get (&edge<T>::_new_dz, G));
    dp.property ("new_phi_slope", boost::get (&edge<T>::_new_PhiSlope, G));
    dp.property ("new_r_phi_slope", boost::get (&edge<T>::_new_rPhiSlope, G));
  }

  //node
  dp.property ("r", boost::get (&vertex<T>::_r, G));
  dp.property ("phi", boost::get (&vertex<T>::_phi, G));
  dp.property ("z", boost::get (&vertex<T>::_z, G));
  if (extra_features) {
    dp.property ("eta", boost::get (&vertex<T>::_eta, G));
    dp.property ("r_cluster1", boost::get (&vertex<T>::_r_cluster1, G));
    dp.property ("phi_cluster1", boost::get (&vertex<T>::_phi_cluster1, G));
    dp.property ("z_cluster1", boost::get (&vertex<T>::_z_cluster1, G));
    dp.property ("eta_cluster1", boost::get (&vertex<T>::_eta_cluster1, G));
    dp.property ("r_cluster2", boost::get (&vertex<T>::_r_cluster2, G));
    dp.property ("phi_cluster2", boost::get (&vertex<T>::_phi_cluster2, G));
    dp.property ("z_cluster2", boost::get (&vertex<T>::_z_cluster2, G));
    dp.property ("eta_cluster2", boost::get (&vertex<T>::_eta_cluster2, G));
  }
  dp.property ("hit_id", boost::get (&vertex<T>::_hit_id, G));

  std::string filename="/event" + event_id + "_INPUT.txt";
  std::string path_SaveGraphs = output_dir + filename;
  std::fstream outGraph (path_SaveGraphs.c_str(), std::ios::out);
  boost::write_graphml (outGraph, G, dp, true);
  outGraph.close();
}


//--------------------------------------------------------------------------------------------------------------
template <class T>
void graph<T>::save_npz (const std::string& event_id, const std::string& output_dir, const bool& strip_hit_pair)
//--------------------------------------------------------------------------------------------------------------
{
  Py_Initialize();
  boost::python::numpy::initialize();

  //--------------
  // node features
  //--------------
  int nb_nodes = boost::num_vertices (G);
  boost::python::numpy::dtype dtype = boost::python::numpy::dtype::get_builtin<T>();
  boost::tuple <T, T, T, T> *node_feat = new boost::tuple <T, T, T, T> [nb_nodes];
  boost::tuple <T, T, T, T> *node_cluster1_feat = new boost::tuple <T, T, T, T> [nb_nodes];
  boost::tuple <T, T, T, T> *node_cluster2_feat = new boost::tuple <T, T, T, T> [nb_nodes];
  int ind = 0;
  vertex_iterator<T> vi, vi_end;

  for (boost::tie (vi, vi_end) = boost::vertices(G); vi != vi_end; ++vi, ind++) {
    node_feat[ind] = boost::tuple <T, T, T, T> (G[*vi].r(), G[*vi].phi(), G[*vi].z(), G[*vi].eta());
    node_cluster1_feat[ind] = boost::tuple <T, T, T, T> (G[*vi].r_cluster1(), G[*vi].phi_cluster1(), G[*vi].z_cluster1(), G[*vi].eta_cluster1());
    node_cluster2_feat[ind] = boost::tuple <T, T, T, T> (G[*vi].r_cluster2(), G[*vi].phi_cluster2(), G[*vi].z_cluster2(), G[*vi].eta_cluster2());
  }

  boost::python::numpy::ndarray nd_nodes = boost::python::numpy::from_data (node_feat, dtype, boost::python::make_tuple (nb_nodes, 4), boost::python::make_tuple (4*sizeof(T), sizeof(T)), boost::python::object());
  boost::python::numpy::ndarray nd_cluster1_nodes = boost::python::numpy::from_data (node_cluster1_feat, dtype, boost::python::make_tuple (nb_nodes, 4), boost::python::make_tuple (4*sizeof(T), sizeof(T)), boost::python::object());
  boost::python::numpy::ndarray nd_cluster2_nodes = boost::python::numpy::from_data (node_cluster2_feat, dtype, boost::python::make_tuple (nb_nodes, 4), boost::python::make_tuple (4*sizeof(T), sizeof(T)), boost::python::object());

  // -------------------------------------
  // edge features / senders and receivers
  //--------------------------------------
  int nb_edges = boost::num_edges (G);
  boost::tuple <T, T, T, T, T, T> *edge_feat = new boost::tuple <T, T, T, T, T, T> [nb_edges];
  int* senders = new int [nb_edges];
  int* receivers = new int [nb_edges];
  ind = 0;
  edge_iterator<T> ei, ei_end;

  for (boost::tie(ei, ei_end) = boost::edges(G); ei != ei_end; ++ei, ind++) {
    edge_feat[ind] = boost::tuple <T, T, T, T, T, T> (G[*ei].dEta(), G[*ei].dPhi(), G[*ei].dr(), G[*ei].dz(), G[*ei].phi_slope(), G[*ei].r_phi_slope());
    receivers[ind] = boost::target (*ei, G);
    senders[ind] = boost::source (*ei, G);
  }

  boost::python::numpy::ndarray nd_edges = boost::python::numpy::from_data (edge_feat, dtype, boost::python::make_tuple (nb_edges, 6), boost::python::make_tuple (6*sizeof(T), sizeof(T)), boost::python::object());
  boost::python::numpy::dtype dtype2 = boost::python::numpy::dtype::get_builtin<int>();
  boost::python::numpy::ndarray nd_receivers = boost::python::numpy::from_data (receivers, dtype2, boost::python::make_tuple (nb_edges), boost::python::make_tuple (sizeof(int)), boost::python::object());
  boost::python::numpy::ndarray nd_senders = boost::python::numpy::from_data (senders, dtype2, boost::python::make_tuple (nb_edges), boost::python::make_tuple (sizeof(int)), boost::python::object());

  // globals
  float glob=0;
  boost::python::numpy::ndarray ndT = boost::python::numpy::from_data (&glob, dtype, boost::python::make_tuple (1), boost::python::make_tuple (sizeof(T)), boost::python::object());

  //----------------------------
  // Write out graph to NPZ file
  //----------------------------
  std::cout << "Writing graph to NPZ file ..." << std::endl;
  boost::python::object my_python_class_module = boost::python::import ("numpy");
  // Retrieve the main module's namespace
  boost::python::object global (my_python_class_module.attr ("__dict__"));
  boost::python::dict kw;
  kw ["nodes"] = nd_nodes;
  kw ["nodes_cluster1"] = nd_cluster1_nodes;
  kw ["nodes_cluster2"] = nd_cluster2_nodes;
  kw ["edges"] = nd_edges;
  kw["receivers"] = nd_receivers;
  kw["senders"] = nd_senders;
 
  boost::tuple <T, T, T, T, T, T, T> *strip_hit_feat = new boost::tuple <T, T, T, T, T, T, T> [nb_edges];
  boost::tuple <T, T, T, T, T, T> *new_edge_feat = new boost::tuple <T, T, T, T, T, T> [nb_edges];

  if (strip_hit_pair) {
    // ------------------
    // strip hit features
    //-------------------
    ind = 0;

    for (boost::tie(ei, ei_end) = boost::edges(G); ei != ei_end; ++ei, ind++) {
      strip_hit_feat[ind] = boost::tuple <T, T, T, T, T, T, T> (G[*ei].new_hit_in_x(), G[*ei].new_hit_in_y(), G[*ei].new_hit_in_z(), G[*ei].new_hit_out_x(), G[*ei].new_hit_out_y(), G[*ei].new_hit_out_z(), (T) G[*ei].pass_strip_range());
      new_edge_feat[ind] = boost::tuple <T, T, T, T, T, T> (G[*ei].new_dEta(), G[*ei].new_dPhi(), G[*ei].new_dr(), G[*ei].new_dz(), G[*ei].new_phi_slope(), G[*ei].new_r_phi_slope());
    }

    boost::python::numpy::ndarray nd_strip_hits = boost::python::numpy::from_data (strip_hit_feat, dtype, boost::python::make_tuple (nb_edges, 7), boost::python::make_tuple (7*sizeof(T), sizeof(T)), boost::python::object());
    boost::python::numpy::ndarray nd_new_edges = boost::python::numpy::from_data (new_edge_feat, dtype, boost::python::make_tuple (nb_edges, 6), boost::python::make_tuple (6*sizeof(T), sizeof(T)), boost::python::object());
    kw["clusters"] = nd_strip_hits;
    kw["new_edges"] = nd_new_edges;
   }

  kw["globals"] = ndT;
  kw ["n_node"] = nb_nodes;
  kw ["n_edge"] = nb_edges;
  std::string filename="/event" + event_id + "_INPUT.npz";
  std::string path_SaveGraphs = output_dir + filename;
  boost::python::object str_obj = boost::python::str (path_SaveGraphs.c_str());
  boost::python::tuple t = make_tuple (str_obj);
  my_python_class_module.attr ("savez_compressed") (*t, **kw);
  std::cout << "... done." << std::endl;
  delete [] node_feat;
  delete [] node_cluster1_feat;
  delete[] node_cluster2_feat;
  delete [] edge_feat;
  delete [] strip_hit_feat;
  delete [] new_edge_feat;
  delete [] senders;
  delete[] receivers;
}

//--------------------------------------------------------------------------------------------------------------
template <class T>
void graph<T>::save_pyg (const std::string& event_id, const std::string& output_dir, const bool& strip_hit_pair)
//--------------------------------------------------------------------------------------------------------------
{
//  torch::FloatTensor tensor = torch::rand({2, 3});
  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << "TENSOR = "<< tensor << std::endl;

//  torch::jit::script::Module tensors = torch::jit::load("x.pth");
//    c10::IValue iv = tensors.attr("x");
//    torch::Tensor ts = iv.toTensor();
//    std::cout << ts;  torch::jit::save("x.pyg");

  torch::jit::script::Module model;
  torch::save (tensor, "cppSavedJitModule.pt");
//  try {
//    model = torch::jit::load("event000000002.pyg");
//  } catch (const c10::Error &e) {
//    std::cerr << "error loading the model\n";
//    return -1;
//  }
//
//  auto x = torch::randn({5, 32});

  //auto edge_index = torch::tensor({
  //    {0, 1, 1, 2, 2, 3, 3, 4},
  //    {1, 0, 2, 1, 3, 2, 4, 3},
//  });

//auto pickled = torch::pickle_save(tensor);
//std::ofstream fout("input.pt", std::ios::out | std::ios::binary);
//fout.write(pickled.data(), pickled.size());
//fout.close();
//
//  std::vector<torch::jit::IValue> inputs;
//  inputs.push_back(x);
//  inputs.push_back(edge_index);

//  auto out = model.forward(inputs).toTensor();
//  std::cout << "output tensor shape: " << out.sizes() << std::endl;
}

//--------------------------------------------------------------------------------------------------------------
template <class T>
void graph<T>::save_csv (const std::string& event_id, const std::string& output_dir, const bool& strip_hit_pair)
//--------------------------------------------------------------------------------------------------------------
{
  // save nodes
  std::string filename="/event" + event_id + "_vertices.csv";
  std::string path_SaveGraphs = output_dir + filename;
  std::ofstream file_vertices (path_SaveGraphs);
  if (file_vertices.fail()) throw std::invalid_argument ("Cannot open file " + filename);

  std::multimap<uint64_t, std::vector<T>> vertices;
  vertex_iterator<T> vi, vi_end;
  for (boost::tie (vi, vi_end) = boost::vertices(G); vi != vi_end; ++vi)
    vertices.insert (std::pair<uint64_t, std::vector<T>> (G[*vi].hit_id(), {G[*vi].r(), G[*vi].z(), G[*vi].phi()}));

  file_vertices << "hit_id,R,z,phi \n";
  for (std::pair<uint64_t, std::vector<T>> vertex : vertices)
    file_vertices << vertex.first << "," << vertex.second[0] << "," << vertex.second[1] << "," << vertex.second[2] << std::endl;

  file_vertices.close ();

  // save edges
  filename = "/event" + event_id + "_edges.csv";
  path_SaveGraphs = output_dir + filename;
  std::ofstream file_edges (path_SaveGraphs);
  if (file_vertices.fail()) throw std::invalid_argument ("Cannot open file " + filename);

  std::multimap<std::vector<uint64_t>, std::vector<T>> edges;
  edge_iterator<T> ei, ei_end;

  for (boost::tie(ei, ei_end) = boost::edges(G); ei != ei_end; ++ei) {
    vertex_type<T> source = boost::source (*ei, G);
    uint64_t source_vertex = G[source].hit_id();
    vertex_type<T> target = boost::target (*ei, G);
    uint64_t target_vertex = G[target].hit_id();
//    edges.insert (std::pair<std::vector<uint64_t>, std::vector<T>> ({G[boost::source (*ei, G)].hit_id(), G[boost::target (*ei, G)].hit_id()}, {G[*ei].dr(), G[*ei].dz(), G[*ei].dEta(), G[*ei].dPhi()}));
    edges.insert (std::pair<std::vector<uint64_t>, std::vector<T>> ({source_vertex, target_vertex}, {G[*ei].dr(), G[*ei].dz(), G[*ei].dEta(), G[*ei].dPhi()}));
  }

  file_edges << "ivertex,overtex,dR,dz,deta,dphi \n";
  for (std::pair<std::vector<uint64_t>, std::vector<T>> edge : edges)
    file_edges << edge.first[0] << "," << edge.first[1] << "," << edge.second[0] << "," << edge.second[1] << "," << edge.second[2] << "," << edge.second[3] << std::endl;
  
  file_edges.close();
}
