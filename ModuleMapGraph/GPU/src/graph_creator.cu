/***************************************************************
 * Tracking project library - L2IT
 * Trace reconstruction in LHC
 * copyright © 2024 COLLARD Christophe
 * copyright © 2024 Centre National de la Recherche Scientifique
 * copyright © 2024 Laboratoire des 2 Infinis de Toulouse (L2IT)
 ***************************************************************/

#include "graph_creator.cuh"

//=====GPU functions for graph creator=======================================

__global__ void print_value (int* vector, int pos)
{
  printf ("vector value at %d = %d \n", pos, vector[pos]);
}

__global__ void print (int* vector, int size)
{
  for (int i=0; i<size; i++)
    printf ("%d ", vector[i]);
}

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <class T>
__global__ void edge_features (T *dR, T *dz, T *r_phi_slope, int *M1_SP, int *M2_SP, int* vertices_sum, T *R, T *z, T *phi_slope, data_type2 pi, T max, int nb_edges)
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb_edges) return;

  int SP1 = M1_SP[i];
  int SP2 = M2_SP[i];
  T r_sum = 0.5 * (R[SP1] + R[SP2]);
  dR[i] = R[SP2] - R[SP1];
  dz[i] = z[SP2] - z[SP1];
  M1_SP[i] = vertices_sum[SP1];
  M2_SP[i] = vertices_sum[SP2];

  if (dR)
    r_phi_slope[i] = r_sum * phi_slope[i];
  else
    r_phi_slope[i] = phi_slope[i];
}

//=====Private methods for graph creator=====================================

//=====Public methods for graph creator======================================

//--------------------------------------------------------------------------------------
template <class T>
CUDA_graph_creator<T>::CUDA_graph_creator (boost::program_options::variables_map &po_vm)
//--------------------------------------------------------------------------------------
{
  _blocks = po_vm["gpu-nb-blocks"].as<int>();
  _events_dir = po_vm["input-dir"].as<std::string>() + "/";
  _output_dir = po_vm["output-dir"].as<std::string>();
  _true_graph = po_vm["give-true-graph"].as<bool>();
  _module_map_dir = po_vm["input-module-map"].template as<std::string>();
  _save_graph_graphml = po_vm["save-graph-on-disk-graphml"].as<bool>();
  _save_graph_npz = po_vm["save-graph-on-disk-npz"].as<bool>();
  _save_graph_pyg = po_vm["save-graph-on-disk-pyg"].as<bool>();
  _save_graph_csv = po_vm["save-graph-on-disk-csv"].as<bool>();
  _strip_hit_pair = po_vm["strip-hit-pair"].as<bool>();
  _extra_features = po_vm["extra-features"].as<bool>();

  if (_strip_hit_pair)
  {
    _strip_module_DB = strip_module_DB<data_type2>(po_vm["input-strip-modules"].as<std::string>());
    _extra_features = true;
  }

  if (_module_map_dir.empty())
    throw std::invalid_argument ("Missing Module Map");

  size_t f, t;
  cudaDeviceReset();
  cudaSetDevice(1);
  cudaMemGetInfo(&f, &t);
  std::cout << "CUDA infos: free memory " << f/pow(1024,3) << " Go / " << t/pow(1024,3) << "Go" << std::endl;

  module_map_triplet<T> module_map_triplet;
  module_map_triplet.read_TTree(_module_map_dir.c_str());
  if (!module_map_triplet)
    throw std::runtime_error("Cannot retrieve ModuleMap from " + _module_map_dir);

  _cuda_module_map_doublet = CUDA_module_map_doublet<T> (module_map_triplet);
  _cuda_module_map_doublet.HostToDevice();
  _cuda_module_map_triplet = CUDA_module_map_triplet<T> (module_map_triplet);
  _cuda_module_map_triplet.HostToDevice();

  std::cout << "# of modules = " << module_map_triplet.module_map().size() << std::endl;

  // log all hits events filenames
  std::string log_hits_filenames = "events-hits.txt";
  std::string cmd_hits = "ls " + _events_dir + "|grep event |grep hits.* > " + log_hits_filenames;
  system(cmd_hits.c_str());
  cmd_hits = "ls " + _events_dir + "|grep event |grep truth.* >> " + log_hits_filenames;
  system(cmd_hits.c_str());

  // log all particles events filenames
  std::string log_particles_filenames = "events-particles.txt";
  std::string cmd_particles = "ls " + _events_dir + "|grep event |grep particles.* > " + log_particles_filenames;
  system (cmd_particles.c_str());

  // open files containing events filenames
  std::ifstream h_file(log_hits_filenames);
  if (h_file.fail())
    throw std::invalid_argument("Cannot open file " + log_hits_filenames);
  std::ifstream p_file(log_particles_filenames);
  if (p_file.fail())
    throw std::invalid_argument("Cannot open file " + log_particles_filenames);

  std::string hits_filename, particles_filename;
  h_file >> hits_filename;
  p_file >> particles_filename;

  for (; !h_file.eof() && !p_file.eof();)
  {
    build (hits_filename, particles_filename);

    h_file >> hits_filename;
    p_file >> particles_filename;
  }

  h_file.close();
  p_file.close();
}

//-------------------------------------------------------------------------------------------
template <class T>
void CUDA_graph_creator<T>::build (std::string hits_filename, std::string particles_filename)
//-------------------------------------------------------------------------------------------
{
  clock_t start, end;
  clock_t start_main, end_main;
  clock_t start_cpy, end_cpy;
  clock_t start_alloc, end_alloc;
  clock_t start_deco, end_deco;
  clock_t start_doublets, end_doublets;
  clock_t start_sum, end_sum;
  clock_t start_reduction, end_reduction;
  clock_t start_geocuts, end_geocuts;
  clock_t start_triplets, end_triplets;
  clock_t start_free, end_free;
  clock_t start_memcpy, end_memcpy;
  start_main = start = clock();

  std::vector<T> _R, _z, _eta, _phi;
  std::vector<T> _dR, _dz, _deta, _dphi;
  std::vector<int> _M1_hits, _M2_hits; // edges hit M1 -> M2

  std::string event_id = boost::regex_replace(hits_filename, boost::regex("[^0-9]*([0-9]+).*"), std::string("\\1"));
  if (event_id != boost::regex_replace(particles_filename, boost::regex("[^0-9]*([0-9]+).*"), std::string("\\1")))
    throw std::invalid_argument("hits and particules file are not part of the same event");

  std::cout << green << "event location " << _events_dir + hits_filename << reset;

  // load event hits
  hits<T> hit_pts(_true_graph, _extra_features);
  TTree_hits<T> TThits (_true_graph, _extra_features);
  std::filesystem::path hits_path = hits_filename;
  if (hits_path.extension() == ".csv")
  {
    hit_pts.read_csv (_events_dir, hits_filename);
    TThits = hit_pts;
  }
  else if (hits_path.extension() == ".root")
    TThits.read (_events_dir + hits_filename);
  else
    throw std::invalid_argument("unknow extension for hit file: " + hits_filename);

  // load event particles
  particles<T> p;
  TTree_particles<T> TTp;
  std::filesystem::path particles_path = particles_filename;
  if (particles_path.extension() == ".csv")
  {
    p.read_csv(_events_dir, particles_filename);
    TTp = p;
  }
  else if (particles_path.extension() == ".root")
    TTp.read(_events_dir + particles_filename);
  else
    throw std::invalid_argument("unknow extension for particles file: " + particles_filename);

  CUDA_TTree_hits<T> cuda_TThits (TThits, _cuda_module_map_doublet.module_map());
  graph<T> G;

  cudaDeviceSynchronize ();
  start = clock();

  dim3 block_dim = _blocks;
  int max_edges = 3000;

  // ----------------
  // copy hits to GPU
  //-----------------

  start_cpy = clock ();
  cuda_TThits.HostToDevice();
  end_cpy = clock ();

  // ------------------------------
  // compute hits decoration values
  // ------------------------------

  start_deco = clock ();
  dim3 grid_dim = ((cuda_TThits.size() + block_dim.x - 1) / block_dim.x);
  TTree_hits_constants<<<grid_dim,block_dim>>> (cuda_TThits.size(), cuda_TThits.cuda_x(), cuda_TThits.cuda_y(), cuda_TThits.cuda_z(), cuda_TThits.cuda_R(), cuda_TThits.cuda_eta(), cuda_TThits.cuda_phi());
  cudaDeviceSynchronize ();
  end_deco = clock ();

  // ----------------------------------
  // memory allocation for hits + edges
  // ----------------------------------
  start_alloc = clock ();
  int nb_doublets = _cuda_module_map_doublet.size();
  int *cuda_M1_hits, *cuda_M2_hits;
  cudaMalloc (&cuda_M1_hits, nb_doublets * max_edges * sizeof(int));
  cudaMalloc (&cuda_M2_hits, nb_doublets * max_edges * sizeof(int));

  int *cuda_nb_edges;
  cudaMalloc (&cuda_nb_edges, nb_doublets * sizeof(int));
  end_alloc = clock ();

  // ---------------------------------------------
  // loop over module doublets to kill connections
  // ---------------------------------------------

  start_doublets = clock();
  grid_dim = ((nb_doublets + block_dim.x - 1) / block_dim.x);
  doublet_cuts<T><<<grid_dim,block_dim>>> (nb_doublets, _cuda_module_map_doublet.cuda_module1(), _cuda_module_map_doublet.cuda_module2(), cuda_TThits.cuda_R(), cuda_TThits.cuda_z(), cuda_TThits.cuda_eta(), cuda_TThits.cuda_phi(),
                                          _cuda_module_map_doublet.cuda_z0_min(), _cuda_module_map_doublet.cuda_z0_max(), _cuda_module_map_doublet.cuda_deta_min(), _cuda_module_map_doublet.cuda_deta_max(),
                                          _cuda_module_map_doublet.cuda_phi_slope_min(), _cuda_module_map_doublet.cuda_phi_slope_max(), _cuda_module_map_doublet.cuda_dphi_min(), _cuda_module_map_doublet.cuda_dphi_max(),
                                          cuda_TThits.cuda_hit_indice(), TMath::Pi(), std::numeric_limits<T>::max(), cuda_M1_hits, cuda_M2_hits, cuda_nb_edges, max_edges);

  cudaDeviceSynchronize ();
  end_doublets = clock();

  //----------------
  // sum nb of edges
  //----------------

  start_sum = clock();

  int *cuda_edge_sum;
  cudaMalloc (&cuda_edge_sum, (nb_doublets+1) * sizeof(int));
  scan (cuda_edge_sum, cuda_nb_edges, _blocks, nb_doublets);

  //---------------------------------------------
  // reduce nb of hits and sort by hid id M2 hits
  //---------------------------------------------

  int nb_doublet_edges;
  cudaMemcpy (&nb_doublet_edges, &(cuda_edge_sum[nb_doublets]), sizeof(int), cudaMemcpyDeviceToHost);
//  std::cout << "nb hits after doublet cuts = " << nb_doublet_edges << std::endl;

  int *cuda_reduced_M1_hits, *cuda_reduced_M2_hits;
  cudaMalloc (&cuda_reduced_M1_hits, nb_doublet_edges * sizeof(int));
  cudaMalloc (&cuda_reduced_M2_hits, nb_doublet_edges * sizeof(int));
  compact_stream<<<grid_dim,block_dim>>> (cuda_reduced_M1_hits, cuda_M1_hits, cuda_edge_sum, max_edges, nb_doublets);
  compact_stream<<<grid_dim,block_dim>>> (cuda_reduced_M2_hits, cuda_M2_hits, cuda_edge_sum, max_edges, nb_doublets);

  int *cuda_sorted_M2_hits;
  cudaMalloc (&cuda_sorted_M2_hits, nb_doublet_edges * sizeof(int));
  grid_dim = ((nb_doublet_edges + block_dim.x - 1) / block_dim.x);
  init_vector<<<grid_dim,block_dim>>> (cuda_sorted_M2_hits, nb_doublet_edges);
  grid_dim = ((nb_doublets + block_dim.x - 1) / block_dim.x);
  partial_quick_sort<<<grid_dim,block_dim>>> (cuda_sorted_M2_hits, cuda_reduced_M2_hits, cuda_edge_sum, nb_doublets);

  end_sum = clock();

  // -----------------------------
  // build doublets geometric cuts
  // -----------------------------

  start_geocuts = clock();

  T *cuda_z0, *cuda_phi_slope, *cuda_deta, *cuda_dphi;
  cudaMalloc (&cuda_z0, nb_doublet_edges * sizeof(T));
  cudaMalloc (&cuda_phi_slope, nb_doublet_edges * sizeof(T));
  cudaMalloc (&cuda_deta, nb_doublet_edges * sizeof(T));
  cudaMalloc (&cuda_dphi, nb_doublet_edges * sizeof(T));
  grid_dim = ((nb_doublet_edges + block_dim.x - 1) / block_dim.x);
  hits_geometric_cuts<<<grid_dim,block_dim>>> (cuda_z0, cuda_phi_slope, cuda_deta, cuda_dphi, cuda_reduced_M1_hits, cuda_reduced_M2_hits, cuda_TThits.cuda_R(), cuda_TThits.cuda_z(), cuda_TThits.cuda_eta(), cuda_TThits.cuda_phi(), TMath::Pi(), std::numeric_limits<T>::max(), nb_doublet_edges);

  int *cuda_edge_tag;
  cudaMalloc (&cuda_edge_tag, nb_doublet_edges * sizeof(int));
  cudaMemset (cuda_edge_tag, 0, nb_doublet_edges * sizeof(int));
//  print_value <<<1,1>>> (cuda_edge_sum, nb_doublets);

  cudaDeviceSynchronize();
  end_geocuts = clock();

  cudaDeviceSynchronize();
  end_geocuts = clock();

  cudaDeviceSynchronize();
  end_geocuts = clock();

  // -------------------------
  // loop over module triplets
  // -------------------------

  start_triplets = clock();
  int nb_triplets = _cuda_module_map_triplet.size();
  grid_dim = ((nb_triplets + block_dim.x - 1) / block_dim.x);
  cudaMemset (cuda_TThits.cuda_vertices(), 0, cuda_TThits.size() * sizeof(int));

  triplet_cuts<T><<<grid_dim,block_dim>>> (nb_triplets, _cuda_module_map_triplet.cuda_module12_map(), _cuda_module_map_triplet.cuda_module23_map(), cuda_TThits.cuda_x(), cuda_TThits.cuda_y(), cuda_TThits.cuda_z(), cuda_TThits.cuda_R(),
                                          cuda_z0, cuda_phi_slope, cuda_deta, cuda_dphi,
                                          _cuda_module_map_triplet.module12().cuda_z0_min(), _cuda_module_map_triplet.module12().cuda_z0_max(), _cuda_module_map_triplet.module12().cuda_deta_min(), _cuda_module_map_triplet.module12().cuda_deta_max(),
                                          _cuda_module_map_triplet.module12().cuda_phi_slope_min(), _cuda_module_map_triplet.module12().cuda_phi_slope_max(), _cuda_module_map_triplet.module12().cuda_dphi_min(), _cuda_module_map_triplet.module12().cuda_dphi_max(),
                                          _cuda_module_map_triplet.module23().cuda_z0_min(), _cuda_module_map_triplet.module23().cuda_z0_max(), _cuda_module_map_triplet.module23().cuda_deta_min(), _cuda_module_map_triplet.module23().cuda_deta_max(),
                                          _cuda_module_map_triplet.module23().cuda_phi_slope_min(), _cuda_module_map_triplet.module23().cuda_phi_slope_max(), _cuda_module_map_triplet.module23().cuda_dphi_min(), _cuda_module_map_triplet.module23().cuda_dphi_max(),
                                          _cuda_module_map_triplet.cuda_diff_dydx_min(), _cuda_module_map_triplet.cuda_diff_dydx_max(), _cuda_module_map_triplet.cuda_diff_dzdr_min(), _cuda_module_map_triplet.cuda_diff_dzdr_max(),
                                          TMath::Pi(), std::numeric_limits<T>::max(), cuda_reduced_M1_hits, cuda_reduced_M2_hits, cuda_sorted_M2_hits, cuda_edge_sum, cuda_TThits.cuda_vertices(), cuda_edge_tag);

  cudaDeviceSynchronize();
  end_triplets = clock();

  //----------------
  // edges reduction
  //----------------

  start_reduction = clock();

  int *cuda_graph_edges_sum;
  cudaMalloc (&cuda_graph_edges_sum, (nb_doublet_edges+1) * sizeof(int));
  scan (cuda_graph_edges_sum, cuda_edge_tag, _blocks, nb_doublet_edges);
  int nb_graph_edges;

  cudaMemcpy (&nb_graph_edges, &(cuda_graph_edges_sum[nb_doublet_edges]), sizeof(int), cudaMemcpyDeviceToHost);

  int *cuda_graph_M1_hits, *cuda_graph_M2_hits;
  T *cuda_reduced_dR, *cuda_reduced_dz, *cuda_reduced_deta, *cuda_reduced_dphi;
  T *cuda_graph_dR, *cuda_graph_dz, *cuda_graph_deta, *cuda_graph_dphi, *cuda_graph_phi_slope, *cuda_graph_r_phi_slope;
  cudaMalloc (&cuda_graph_M1_hits, nb_graph_edges * sizeof(int));
  cudaMalloc (&cuda_graph_M2_hits, nb_graph_edges * sizeof(int));
  cudaMalloc (&cuda_graph_deta, nb_graph_edges * sizeof(T));
  cudaMalloc (&cuda_graph_dphi, nb_graph_edges * sizeof(T));
  cudaMalloc (&cuda_graph_phi_slope, nb_graph_edges * sizeof(T));

  compact_stream<<<grid_dim,block_dim>>> (cuda_graph_M1_hits, cuda_reduced_M1_hits, cuda_edge_tag, cuda_graph_edges_sum, nb_doublet_edges);
  compact_stream<<<grid_dim,block_dim>>> (cuda_graph_M2_hits, cuda_reduced_M2_hits, cuda_edge_tag, cuda_graph_edges_sum, nb_doublet_edges);
  compact_stream<<<grid_dim,block_dim>>> (cuda_graph_deta, cuda_deta, cuda_edge_tag, cuda_graph_edges_sum, nb_doublet_edges);
  compact_stream<<<grid_dim,block_dim>>> (cuda_graph_dphi, cuda_dphi, cuda_edge_tag, cuda_graph_edges_sum, nb_doublet_edges);
  compact_stream<<<grid_dim,block_dim>>> (cuda_graph_phi_slope, cuda_phi_slope, cuda_edge_tag, cuda_graph_edges_sum, nb_doublet_edges);

  //----------------
  // nodes reduction
  //----------------

  int nb_hits = cuda_TThits.size();
  int *cuda_graph_vertices_sum;
  cudaMalloc (&cuda_graph_vertices_sum, (nb_hits+1) * sizeof(int));
  scan (cuda_graph_vertices_sum, cuda_TThits.cuda_vertices(), _blocks, nb_hits);
  int nb_graph_hits;
  cudaMemcpy (&nb_graph_hits, &(cuda_graph_vertices_sum[nb_hits]), sizeof(int), cudaMemcpyDeviceToHost);

  uint64_t *cuda_graph_hitsID;
  T *cuda_graph_R, *cuda_graph_z, *cuda_graph_eta, *cuda_graph_phi;
  cudaMalloc (&cuda_graph_hitsID, nb_graph_hits * sizeof(uint64_t));
  cudaMalloc (&cuda_graph_R, nb_graph_hits * sizeof(T));
  cudaMalloc (&cuda_graph_z, nb_graph_hits * sizeof(T));
  cudaMalloc (&cuda_graph_eta, nb_graph_hits * sizeof(T));
  cudaMalloc (&cuda_graph_phi, nb_graph_hits * sizeof(T));

  grid_dim = ((nb_hits + block_dim.x - 1) / block_dim.x);
  compact_stream<<<grid_dim,block_dim>>> (cuda_graph_hitsID, cuda_TThits.cuda_hit_id(), cuda_TThits.cuda_vertices(), cuda_graph_vertices_sum, nb_hits);
  compact_stream<<<grid_dim,block_dim>>> (cuda_graph_R, cuda_TThits.cuda_R(), cuda_TThits.cuda_vertices(), cuda_graph_vertices_sum, nb_hits);
  compact_stream<<<grid_dim,block_dim>>> (cuda_graph_z, cuda_TThits.cuda_z(), cuda_TThits.cuda_vertices(), cuda_graph_vertices_sum, nb_hits);

  cudaMalloc (&cuda_graph_dR, nb_graph_edges * sizeof(T));
  cudaMalloc (&cuda_graph_dz, nb_graph_edges * sizeof(T));
  cudaMalloc (&cuda_graph_r_phi_slope, nb_graph_edges * sizeof(T));
  grid_dim = ((nb_graph_edges + block_dim.x - 1) / block_dim.x);
  edge_features<<<grid_dim,block_dim>>> (cuda_graph_dR, cuda_graph_dz, cuda_graph_r_phi_slope, cuda_graph_M1_hits, cuda_graph_M2_hits, cuda_graph_vertices_sum, cuda_TThits.cuda_R(), cuda_TThits.cuda_z(), cuda_graph_phi_slope, TMath::Pi(), std::numeric_limits<T>::max(), nb_graph_edges);

  cudaDeviceSynchronize();
  end_reduction = clock();

  // --------------------
  // copy data GPU -> CPU
  // --------------------

  start_memcpy = clock();

  std::vector<uint64_t> hitsID (nb_graph_hits);
  std::vector<T> R(nb_graph_hits), z(nb_graph_hits), eta(nb_graph_hits), phi(nb_graph_hits), phi_slope(nb_graph_hits), r_phi_slope(nb_graph_hits);
  cudaMemcpy (hitsID.data(), cuda_graph_hitsID, nb_graph_hits * sizeof(uint64_t), cudaMemcpyDeviceToHost);
  cudaMemcpy (R.data(), cuda_graph_R, nb_graph_hits * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy (z.data(), cuda_graph_z, nb_graph_hits * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy (eta.data(), cuda_graph_eta, nb_graph_hits * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy (phi.data(), cuda_graph_phi, nb_graph_hits * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy (phi_slope.data(), cuda_graph_phi_slope, nb_graph_hits * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy (r_phi_slope.data(), cuda_graph_r_phi_slope, nb_graph_hits * sizeof(T), cudaMemcpyDeviceToHost);

  std::vector<int> M1_hits (nb_graph_edges), M2_hits (nb_graph_edges);

  std::vector<T> dR(nb_graph_edges), dz(nb_graph_edges), deta(nb_graph_edges), dphi(nb_graph_edges);
  cudaMemcpy (M1_hits.data(), cuda_graph_M1_hits, nb_graph_edges * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy (M2_hits.data(), cuda_graph_M2_hits, nb_graph_edges * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy (dR.data(), cuda_graph_dR, nb_graph_edges * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy (dz.data(), cuda_graph_dz, nb_graph_edges * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy (deta.data(), cuda_graph_deta, nb_graph_edges * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy (dphi.data(), cuda_graph_dphi, nb_graph_edges * sizeof(T), cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();
  end_memcpy = clock();

  // get memory info
  size_t free_memory, total_memory;
  cudaMemGetInfo (&free_memory, &total_memory);

/*
  //----------------------------------------------------------------------------------
  // check max_edges is correctly calibrated - comment this part for performance tests
  // do not move this test - must be executed before cudaFree
  std::vector<int> nb_edges (nb_doublets);
  cudaMemcpy (nb_edges.data(), cuda_nb_edges, nb_doublets * sizeof(int), cudaMemcpyDeviceToHost);
  for (int i=0; i<nb_doublets; i++)
    if (nb_edges[i] >= max_edges-2) {
      std::cout << red << "ERROR: element " << i << " increase max edges value - actual value is over " << nb_edges[i] << reset;
      exit(0);
    }
  //----------------------------------------------------------------------------------
*/

  //-------------------
  // free memory on GPU
  //-------------------

  start_free = clock();

  cudaFree (cuda_graph_r_phi_slope);
  cudaFree (cuda_graph_dz);
  cudaFree (cuda_graph_dR);
  cudaFree (cuda_graph_phi_slope);
  cudaFree (cuda_graph_dphi);
  cudaFree (cuda_graph_deta);
  cudaFree (cuda_graph_M2_hits);
  cudaFree (cuda_graph_M1_hits);
  cudaFree (cuda_graph_edges_sum);
  cudaFree (cuda_graph_phi);
  cudaFree (cuda_graph_eta);
  cudaFree (cuda_graph_z);
  cudaFree (cuda_graph_R);
  cudaFree (cuda_graph_hitsID);
  cudaFree (cuda_graph_vertices_sum);
  cudaFree (cuda_edge_tag);
  cudaFree (cuda_dphi);
  cudaFree (cuda_deta);
  cudaFree (cuda_phi_slope);
  cudaFree (cuda_z0);
  cudaFree (cuda_sorted_M2_hits);
  cudaFree (cuda_reduced_M2_hits);
  cudaFree (cuda_reduced_M1_hits);
  cudaFree (cuda_edge_sum);
  cudaFree (cuda_nb_edges);
  cudaFree (cuda_M2_hits);
  cudaFree (cuda_M1_hits);

  cudaDeviceSynchronize();
  end_free = clock();

  for (int i=0; i<nb_graph_hits; i++)
    G.add_vertex (hitsID[i], R[i], z[i], eta[i], phi[i]);

  for (int i=0; i<nb_graph_edges; i++) {
    edge<T> edg (deta[i], dphi[i], dR[i], dz[i], phi_slope[i], r_phi_slope[i]);
//    G.add_edge (cuda_TThits.vertices()[M1_hits[i]], cuda_TThits.vertices()[M2_hits[i]], edg);
    G.add_edge (M1_hits[i], M2_hits[i], edg);
//    G.add_edge (graph_vertices_sum[M1_hits[i]], graph_vertices_sum[M2_hits[i]], edg);
  }

  std::cout << "nb nodes = " << nb_graph_hits << std::endl;
  std::cout << "nb edges after MD cuts = " << nb_doublet_edges << std::endl;
  std::cout << "nb edges after MT cuts = " << nb_graph_edges << std::endl;

  end = clock ();

  std::cout << "cpu time for doublets loop: " << (long double)(end_doublets - start_doublets) / CLOCKS_PER_SEC << std::endl;
  std::cout << cyan << "cpu time for hits copy to GPU memory: " << (long double)(end_cpy - start_cpy) / CLOCKS_PER_SEC << reset;
  std::cout << cyan << "cpu time for hits decorations: " << (long double)(end_deco - start_deco) / CLOCKS_PER_SEC << reset;
  std::cout << cyan << "cpu time for memory allocation: " << (long double)(end_alloc - start_alloc) / CLOCKS_PER_SEC << reset;
  std::cout << cyan << "cpu time for doublets selection: " << (long double)(end_doublets - start_doublets) / CLOCKS_PER_SEC << reset;
  std::cout << cyan << "cpu time for sum edges: " << (long double)(end_sum - start_sum) / CLOCKS_PER_SEC << reset;
  std::cout << cyan << "cpu time for geometric cuts (doublets only): " << (long double)(end_geocuts - start_geocuts) / CLOCKS_PER_SEC << reset;
  std::cout << cyan << "cpu time for triplets selection: " << (long double)(end_triplets - start_triplets) / CLOCKS_PER_SEC << reset;
  std::cout << cyan << "cpu time for reduction: " << (long double)(end_reduction - start_reduction) / CLOCKS_PER_SEC << reset;
  std::cout << cyan << "cpu time to copy data from device to host: " << (long double)(end_memcpy - start_memcpy) / CLOCKS_PER_SEC << reset;
  std::cout << cyan << "cpu time to free memory: " << (long double)(end_free - start_free) / CLOCKS_PER_SEC << reset;
  std::cout << red << "cpu time for event processing (loop only - no i/o - no graphml): " << (long double)(end_free - start) / CLOCKS_PER_SEC << reset;
  std::cout << magenta << "cpu time for global selection (graphml construction): " << (long double)(end - start) / CLOCKS_PER_SEC << reset;
  std::cout << blue << "CUDA infos: used memory " << (total_memory - free_memory)/pow(1024,3) << " Go / " << total_memory/pow(1024,3) << "Go" << reset;

_save_graph_graphml = false; // delete this
  if (_save_graph_graphml) {
    std::cout << "writing output in directory " << _output_dir << std::endl;
    std::cout << "event id = " << event_id << std::endl;
    G.save (event_id, _output_dir, _extra_features);
//        if (_true_graph) G_true.save (event_id, _output_dir);
  }

  std::ofstream outfile;
  outfile.open("excel-data.txt", std::ios_base::app);

  outfile << (long double)(end_free - start) / CLOCKS_PER_SEC << ","
            << nb_graph_edges << ","
            << nb_graph_hits << ","
            << (long double)(end_cpy - start_cpy) / CLOCKS_PER_SEC << ","
            << (long double)(end_deco - start_deco) / CLOCKS_PER_SEC << ","
            << (long double)(end_alloc - start_alloc) / CLOCKS_PER_SEC << ","
            << (long double)(end_doublets - start_doublets) / CLOCKS_PER_SEC << ","
            << (long double)(end_sum - start_sum) / CLOCKS_PER_SEC << ","
            << (long double)(end_triplets - start_triplets) / CLOCKS_PER_SEC << ","
            << (long double)(end_reduction - start_reduction) / CLOCKS_PER_SEC << ","
            << (long double)(end_memcpy - start_memcpy) / CLOCKS_PER_SEC << ","
            << (long double)(end_free - start_free) / CLOCKS_PER_SEC << ","
//            << (total_memory - free_memory)/pow(1024,3)
            << std::endl;
  outfile.close();

/*
      if (_save_graph_npz) {
        G.save_npz (event_id, _output_dir, _extra_features);
        if (_true_graph) G_true.save_npz (event_id, _output_dir);
      }

      if (_save_graph_pyg) {
        G.save_pyg (event_id, _output_dir, _extra_features);
        //if (_true_graph) G_true.save_pyg (event_id, _output_dir);
      }
  */

  if (_save_graph_csv) {
    G.save_csv (event_id, _output_dir, _extra_features);
    //        if (_true_graph) G_true.save (event_id, _output_dir);
  }

  end_main = clock();
  std::cout << "cpu time for GraphCreatorWriterModuleTriplet: " << (long double)(end_main - start_main) / CLOCKS_PER_SEC << std::endl;
}
