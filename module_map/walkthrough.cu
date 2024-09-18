/***************************************************************
 * Tracking project library - L2IT
 * Trace reconstruction in LHC
 * copyright © 2024 COLLARD Christophe
 * copyright © 2024 Centre National de la Recherche Scientifique
 * copyright © 2024 Laboratoire des 2 Infinis de Toulouse (L2IT)
 ***************************************************************/

#include "walkthrough.cuh"


template <class T>
__global__ void print (T* vector, int size)
{
  for (int i=0; i<size; i++)
    printf ("%d ", vector[i]);
}

//=====GPU functions for walk through========================================

__global__ void get ()
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

//  if (i>2 || j>2) return;
//if (i) return;

  printf ("%d,%d \n", i, j);
}

//-------------------------------------------------------
template <class T>
__global__ void init_value (T* vector, T value, int size)
//-------------------------------------------------------
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    vector[i] = value;
}

//-----------------------------------------------------------------------------------------------------------
template <class T>
__global__ void kill_direct_connexions (T *imask, T *omask, int *ivertex, int *overtex, int isize, int osize)
//-----------------------------------------------------------------------------------------------------------
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= isize || j >= osize) return;
  imask[i] = omask[j] = (ivertex[i] != overtex[j]);
}

//--------------------------------------------------------------------------------------------------
template <class T>
__global__ void kill_dead_end_connexions (T *mask, int *ivertex, int *overtex, int isize, int osize)
//--------------------------------------------------------------------------------------------------
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= isize || j >= osize) return;

  if (ivertex[i] == overtex[j])
    mask[j] = true;
}

//-----------------------------------------------------------------
template <class T>
__global__ void edge_cuts (int size, T* weights, T cut, int* edges)
//-----------------------------------------------------------------
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) return;

  edges[i] = (weights[i] > cut);
}

//------------------------------------------------------------------------------------------------------
__global__ void end_vertex (bool* first_vertex, bool* last_vertex, int* ivertex, int* overtex, int size)
//------------------------------------------------------------------------------------------------------
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= size || j >= size) return;

  if (ivertex[i] == overtex[j]) {
    first_vertex[i] = false;
    last_vertex[j] = false;
  }
}

//-----------------------------------------------------------------------------------------------
__global__ void next_vertex (bool *next_vertex, int* ivertex, int* overtex, int isize, int osize)
//-----------------------------------------------------------------------------------------------
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= isize || j >= osize) return;

  if (ivertex[i] == overtex[j])
    next_vertex[j] = true;
}

/*
//--------------------------------------------------------------------------------------------------------------
__global__ void next_vertex (bool* next_vertex, int* ivertex, int* overtex, int nb_vetices, int nb_first_vertex)
//--------------------------------------------------------------------------------------------------------------
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= nb_vertices || j >= nb_first_vertex) return;

  if (ivertex[i] == overtex[j])
    next_vertex[i] = true;
}
*/

//-----------------------------------------------------------------------
//__global__ void select_edges (int size, int nb_vertices, int* ivertex, int* overtex, int* reduced_first_vertex)
__global__ void select_edges (int size, int nb_vertices, int* ivertex, int* overtex, int* reduced_first_vertex, int* edge_range)
//-----------------------------------------------------------------------
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i > size) return;

  for (int k=0; k<nb_vertices && !edge_range[i]; k++)
    if (ivertex[i] == reduced_first_vertex[k])
      edge_range[i] = overtex[i];
}


//=====CPU functions for walk through========================================


//----------------------------------------------------------------------------------
template <class T>
CUDA_walkthrough<T>::CUDA_walkthrough (boost::program_options::variables_map &po_vm)
//----------------------------------------------------------------------------------
{
  _blocks = po_vm["gpu-nb-blocks"].as<int>();
  _events_dir = po_vm["input-dir"].as<std::string>() + "/";
  _output_dir = po_vm["output-dir"].as<std::string>();
  _module_map_dir = po_vm["input-module-map"].template as<std::string>();

  size_t f, t;
  cudaDeviceReset();
  cudaSetDevice(1);
  cudaMemGetInfo(&f, &t);
  std::cout << "CUDA infos: free memory " << f/pow(1024,3) << " Go / " << t/pow(1024,3) << "Go" << std::endl;

  // log all vertices events filenames
  std::string log_vertices_filenames = "events-vertices.txt";
  std::string cmd_vertices = "ls " + _events_dir + "|grep event |grep vertices.* > " + log_vertices_filenames;
  system(cmd_vertices.c_str());

  // log all edges events filenames
  std::string log_edges_filenames = "events-edges.txt";
  std::string cmd_edges = "ls " + _events_dir + "|grep event |grep edges.* > " + log_edges_filenames;
  system(cmd_edges.c_str());

  // open files containing events filenames
  std::ifstream v_file (log_vertices_filenames);
  if (v_file.fail())
    throw std::invalid_argument("Cannot open file " + log_vertices_filenames);

  std::ifstream e_file (log_edges_filenames);
  if (e_file.fail())
    throw std::invalid_argument("Cannot open file " + log_edges_filenames);

  std::string vertices_filename, edges_filename;
  v_file >> vertices_filename;
  e_file >> edges_filename;

  for (; !e_file.eof() && !v_file.eof();) {
    std::cout << "vertices filename = " << vertices_filename << std::endl;
    std::cout << "edges filename = " << edges_filename << std::endl;
    CUDA_graph<T> G (_events_dir, vertices_filename, edges_filename);
    e_file >> edges_filename;

    incoming_vertex(G);
//    cudaFree (_first_vertex);
    std::cout << "end step" << std::endl;
  }

  e_file.close();

  module_map_triplet<T> module_map_triplet;
  module_map_triplet.read_TTree(_module_map_dir.c_str());
  if (!module_map_triplet)
    throw std::runtime_error("Cannot retrieve ModuleMap from " + _module_map_dir);

  _cuda_module_map_doublet = CUDA_module_map_doublet<T> (module_map_triplet);
  _cuda_module_map_doublet.HostToDevice();
  _cuda_module_map_triplet = CUDA_module_map_triplet<T> (module_map_triplet);
  _cuda_module_map_triplet.HostToDevice();


  std::cout << "# of modules = " << module_map_triplet.module_map().size() << std::endl;
}


//----------------------------------------------------------------------------------
template <class T>
void CUDA_walkthrough<T>::incoming_vertex (const CUDA_graph<T>& G)
//----------------------------------------------------------------------------------
{
  clock_t start, end;
  clock_t start_cut, end_cut;
  clock_t start_stream_reduction, end_stream_reduction;
  clock_t start_end_vertex, end_end_vertex;
  clock_t start_last_vertex, end_last_vertex;
  clock_t start_second_vertex, end_second_vertex;
  clock_t start_dead_end_connections, end_dead_end_connections;
  start = clock();

//dim3 bd (32,32);
//int sz = (10 + bd.x - 1) / bd.x + 1;
//dim3 gd (sz,sz);
//std::cout << "sz = " << sz << " / grid dim = " << gd.x << "," << gd.y << " / block dim = " << bd.x << "," << bd.y << std::endl;
//get<<<gd,bd>>> ();

  dim3 block_dim = _blocks;
  dim3 grid_dim = ((G.edges() + block_dim.x - 1) / block_dim.x);

  // -------------------
  // cuts on edge weight (weights are issued by the GNN algorithm)
  // -------------------
  
  start_cut = clock();

  T cut = 0.01;
  int *cuda_edge_mask;
  cudaMalloc (&cuda_edge_mask, G.edges() * sizeof(int));
  edge_cuts<<<grid_dim,block_dim>>> (G.edges(), G.weights(), cut, cuda_edge_mask);

  cudaDeviceSynchronize();
  end_cut = clock ();

  // ----------------------
  // sum nb of edges (scan)
  //-----------------------

  clock_t start_scan = clock();
  int *cuda_edge_mask_sum;
  cudaMalloc (&cuda_edge_mask_sum, (G.edges() + 1) * sizeof(int));
  scan (cuda_edge_mask_sum, cuda_edge_mask, _blocks, G.edges());
  int nb_edges;
  cudaMemcpy (&nb_edges, &(cuda_edge_mask_sum[G.edges()]), sizeof(int), cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();
  clock_t end_scan = clock();

  // ------------------------------------
  // stream reduction of hits and weigths
  // ------------------------------------

  start_stream_reduction = clock();

  int *reduced_vertices_in;
  cudaMalloc (&reduced_vertices_in, nb_edges * sizeof(int));
  compact_stream<int><<<grid_dim,block_dim>>> (reduced_vertices_in, G.vertices_in(), cuda_edge_mask, cuda_edge_mask_sum, G.edges());

  int *reduced_vertices_out;
  cudaMalloc (&reduced_vertices_out, nb_edges * sizeof(int));
  compact_stream<int><<<grid_dim,block_dim>>> (reduced_vertices_out, G.vertices_out(), cuda_edge_mask, cuda_edge_mask_sum, G.edges());

  T *reduced_weights;
  cudaMalloc (&reduced_weights, nb_edges * sizeof(T));
  compact_stream<T><<<grid_dim,block_dim>>> (reduced_weights, G.weights(), cuda_edge_mask, cuda_edge_mask_sum, G.edges());

  cudaDeviceSynchronize();
  end_stream_reduction = clock();

  // ---------------------
  // look for end vertices
  // ---------------------
//fct find end_vertices

  start_end_vertex = clock();

  grid_dim = (nb_edges + _blocks - 1) / _blocks;
  bool *cuda_first_vertex_mask, *cuda_last_vertex_mask;
  cudaMalloc (&cuda_first_vertex_mask, nb_edges * sizeof(bool));
  cudaMalloc (&cuda_last_vertex_mask, nb_edges * sizeof(bool));
  init_value<<<grid_dim,block_dim>>> (cuda_first_vertex_mask, true, nb_edges);
  init_value<<<grid_dim,block_dim>>> (cuda_last_vertex_mask, true, nb_edges);

  int blocks_2d = sqrt (2 * _blocks);
  int size_2d = (nb_edges + blocks_2d - 1) / blocks_2d;
  dim3 block_dim_2d (blocks_2d, blocks_2d);
  dim3 grid_dim_2d (size_2d, size_2d);
  end_vertex<<<grid_dim_2d,block_dim_2d>>> (cuda_first_vertex_mask, cuda_last_vertex_mask, reduced_vertices_in, reduced_vertices_out, nb_edges);

  // merge first and last vertex masks
//fct reduce edges

  bool *cuda_end_vertex_mask;
  cudaMalloc (&cuda_end_vertex_mask, nb_edges * sizeof(int));
  merge_masks<<<grid_dim,block_dim>>> (cuda_end_vertex_mask, cuda_first_vertex_mask, cuda_last_vertex_mask, nb_edges);

  // scan global mask (to reduce end edges)

  int *cuda_end_vertex_mask_sum;
  cudaMalloc (&cuda_end_vertex_mask_sum, (nb_edges+1) * sizeof(int));
  scan (cuda_end_vertex_mask_sum, cuda_end_vertex_mask, _blocks, nb_edges);
  int nb_end_vertices;
  cudaMemcpy (&nb_end_vertices, &(cuda_end_vertex_mask_sum[nb_edges]), sizeof(int), cudaMemcpyDeviceToHost);

  // reduce edges (vertices in and out)

  bool *cuda_other_vertex_mask;
  cudaMalloc (&cuda_other_vertex_mask, nb_edges * sizeof(bool));
  reverse_mask<<<grid_dim,block_dim>>> (cuda_other_vertex_mask, cuda_end_vertex_mask, nb_edges);

  int *cuda_other_vertex_mask_sum;
  cudaMalloc (&cuda_other_vertex_mask_sum, (nb_edges+1) * sizeof(int));
  scan (cuda_other_vertex_mask_sum, cuda_other_vertex_mask, _blocks, nb_edges);

  int nb_other_vertices;
  cudaMemcpy (&nb_other_vertices, &(cuda_other_vertex_mask_sum[nb_edges]), sizeof(int), cudaMemcpyDeviceToHost);

  int *cuda_other_vertex_in, *cuda_other_vertex_out;
  cudaMalloc (&cuda_other_vertex_in, nb_other_vertices * sizeof(int));
  cudaMalloc (&cuda_other_vertex_out, nb_other_vertices * sizeof(int));
  compact_stream<<<grid_dim,block_dim>>> (cuda_other_vertex_in, reduced_vertices_in, cuda_other_vertex_mask, cuda_other_vertex_mask_sum, nb_edges);
  compact_stream<<<grid_dim,block_dim>>> (cuda_other_vertex_out, reduced_vertices_out, cuda_other_vertex_mask, cuda_other_vertex_mask_sum, nb_edges);

  // reduce hit_id list

  bool *cuda_hitID_in_mask;
  cudaMalloc (&cuda_hitID_in_mask, G.vertices() * sizeof (bool));
  cudaMemset (cuda_hitID_in_mask, false, G.vertices() * sizeof(bool));
  dim3 grid_dim_hits = (nb_other_vertices + _blocks - 1) / _blocks;
  tag_mask<<<grid_dim_hits,block_dim>>> (cuda_hitID_in_mask, cuda_other_vertex_in, nb_other_vertices);
  int *cuda_hitID_in_mask_sum;
  cudaMalloc (&cuda_hitID_in_mask_sum, (G.vertices()+1) * sizeof(int));
  scan (cuda_hitID_in_mask_sum, cuda_hitID_in_mask, _blocks, G.vertices());

  int nb_hitsID_in;
  cudaMemcpy (&nb_hitsID_in, &(cuda_hitID_in_mask_sum[G.vertices()]), sizeof(int), cudaMemcpyDeviceToHost);

  int *cuda_other_hits_in;
  cudaMalloc (&cuda_other_hits_in, nb_hitsID_in * sizeof(int));
  grid_dim_hits = (G.vertices() + _blocks - 1) / _blocks;
  compact_stream<<<grid_dim_hits,block_dim>>> (cuda_other_hits_in, G.hit_id(), cuda_hitID_in_mask, cuda_hitID_in_mask_sum, G.vertices());

// fct reduce end_edges (1 fct for first and last edge called twice)
  // reduce end edges

  int *cuda_first_vertex_mask_sum, *cuda_last_vertex_mask_sum;
  cudaMalloc (&cuda_first_vertex_mask_sum, (nb_edges+1) * sizeof(int));
  cudaMalloc (&cuda_last_vertex_mask_sum, (nb_edges+1) * sizeof(int));
  scan (cuda_first_vertex_mask_sum, cuda_first_vertex_mask, _blocks, nb_edges);
  scan (cuda_last_vertex_mask_sum, cuda_last_vertex_mask, _blocks, nb_edges);
  int nb_first_vertices, nb_last_vertices;
  cudaMemcpy (&nb_first_vertices, &(cuda_first_vertex_mask_sum[nb_edges]), sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy (&nb_last_vertices, &(cuda_last_vertex_mask_sum[nb_edges]), sizeof(int), cudaMemcpyDeviceToHost);

  int *cuda_first_edge_in, *cuda_first_edge_out;
  cudaMalloc (&cuda_first_edge_in, nb_first_vertices * sizeof(int));
  cudaMalloc (&cuda_first_edge_out, nb_first_vertices * sizeof(int));
  compact_stream<<<grid_dim,block_dim>>> (cuda_first_edge_in, reduced_vertices_in, cuda_first_vertex_mask, cuda_first_vertex_mask_sum, nb_edges);
  compact_stream<<<grid_dim,block_dim>>> (cuda_first_edge_out, reduced_vertices_out, cuda_first_vertex_mask, cuda_first_vertex_mask_sum, nb_edges);

  int *cuda_last_edge_in, *cuda_last_edge_out;
  cudaMalloc (&cuda_last_edge_in, nb_last_vertices * sizeof(int));
  cudaMalloc (&cuda_last_edge_out, nb_last_vertices * sizeof(int));
  compact_stream<<<grid_dim,block_dim>>> (cuda_last_edge_in, reduced_vertices_in, cuda_last_vertex_mask, cuda_last_vertex_mask_sum, nb_edges);
  compact_stream<<<grid_dim,block_dim>>> (cuda_last_edge_out, reduced_vertices_out, cuda_last_vertex_mask, cuda_last_vertex_mask_sum, nb_edges);

  // kills dead end edges (cam replace cuda_other_vertices with shorter ordered and unique hit list cud_other_hits_in)

  start_dead_end_connections = clock();

  bool *cuda_dead_end_mask;
  cudaMalloc (&cuda_dead_end_mask, nb_first_vertices * sizeof(bool));
  cudaMemset (cuda_dead_end_mask, false, nb_first_vertices * sizeof(bool));
  int sizeo = (nb_first_vertices + blocks_2d - 1) / blocks_2d;
  int sizei = (nb_other_vertices + blocks_2d - 1) / blocks_2d;
  dim3 grid_kill (sizei, sizeo);
  kill_dead_end_connexions<<<grid_kill,block_dim_2d>>> (cuda_dead_end_mask, cuda_other_vertex_in, cuda_first_edge_out, nb_other_vertices, nb_first_vertices);

  int *cuda_dead_end_mask_sum;
  cudaMalloc (&cuda_dead_end_mask_sum, (nb_first_vertices+1) * sizeof(int));
  scan (cuda_dead_end_mask_sum, cuda_dead_end_mask, _blocks, nb_first_vertices);
  int nb_not_dead_end_vertices;
  cudaMemcpy (&nb_not_dead_end_vertices, &(cuda_dead_end_mask_sum[nb_first_vertices]), sizeof(int), cudaMemcpyDeviceToHost);

  int *cuda_first_vertex_out;
  cudaMalloc (&cuda_first_vertex_out, (nb_first_vertices - nb_not_dead_end_vertices) * sizeof(int));
  grid_dim = (nb_first_vertices + _blocks - 1) / _blocks;
  compact_stream <<<grid_dim,block_dim>>> (cuda_first_vertex_out, cuda_first_edge_out, cuda_dead_end_mask, cuda_dead_end_mask_sum, nb_first_vertices);
  nb_first_vertices -= nb_not_dead_end_vertices;

  // extract unique output hit
  bool *cuda_hits_mask;
  cudaMalloc (&cuda_hits_mask, G.vertices() * sizeof(bool));
  cudaMemset (cuda_hits_mask, 0, G.vertices() * sizeof(bool));
  grid_dim = (G.vertices() + _blocks - 1) / _blocks;
//  init_vector<<<grid_dim,block_dim>>> (cuda_sorted_hits, nb_first_vertices);

  // kill direct connexions fist <-> last edge FAUX sauf si connexion unique
//  bool *cuda_first_edge_mask, *cuda_last_edge_mask;
//  cudaMalloc (&cuda_first_edge_mask, nb_first_vertices * sizeof(int));
//  cudaMalloc (&cuda_last_edge_mask, nb_first_vertices * sizeof(int));
//  int sizeo = (nb_first_vertices + blocks - 1) / blocks;
//  int sizei = (nb_last_vertices + blocks - 1) / blocks;
//  dim3 grid_kill (sizei, sizeo);
//  kill_direct_connexions<<<grid_kill,block_dim_2d>>> (cuda_last_edge_mask, cuda_first_edge_mask, cuda_last_edge_in, cuda_first_edge_out, nb_last_vertices, nb_first_vertices);

  cudaDeviceSynchronize();
  end_dead_end_connections = clock();
  end_end_vertex = clock();

  // ---------------------
  // look for 2nd vertices
  // ---------------------

  start_second_vertex = clock();

  int size_vtxo = (nb_other_vertices + blocks_2d - 1) / blocks_2d;
  int size_vtx1 = (nb_first_vertices + blocks_2d - 1) / blocks_2d;
  dim3 grid_dim_2d_vtx2 (size_vtx1, size_vtxo);
  bool *cuda_second_vertex_mask;
  int *cuda_edge_source_location;
  cudaMalloc (&cuda_second_vertex_mask, nb_other_vertices * sizeof(bool));
  // FAUX !!!! plusieurs connexions par noeud
  next_vertex<<<grid_dim_2d_vtx2,block_dim_2d>>> (cuda_second_vertex_mask, cuda_first_edge_out, cuda_other_vertex_in, nb_first_vertices, nb_other_vertices);

  int *cuda_second_vertex_mask_sum;
  cudaMalloc (&cuda_second_vertex_mask_sum, (nb_other_vertices+1) * sizeof(int));
  scan (cuda_second_vertex_mask_sum, cuda_second_vertex_mask, _blocks, nb_other_vertices);
  int nb_second_vertices;
  cudaMemcpy (&nb_second_vertices, &(cuda_second_vertex_mask_sum[nb_other_vertices]), sizeof(int), cudaMemcpyDeviceToHost);


  cudaDeviceSynchronize();
  end_second_vertex = clock();

  // get memory info
  size_t free_memory, total_memory;
  cudaMemGetInfo (&free_memory, &total_memory);

//  cudaFree (_first_vertex);
//  cudaFree (cuda_edge_source_location);
//  cuda_free (cuda_second_vertex_mask);
//  cudaFree (cuda_last_edge_mask);
//  cudaFree (cuda_first_edge_mask);
  cudaFree (cuda_first_vertex_out);
  cudaFree (cuda_dead_end_mask_sum);
  cudaFree (cuda_dead_end_mask);
  cudaFree (cuda_last_edge_out);
  cudaFree (cuda_last_edge_in);
  cudaFree (cuda_first_edge_out);
  cudaFree (cuda_first_edge_in);
  cudaFree (cuda_last_vertex_mask_sum);
  cudaFree (cuda_first_vertex_mask_sum);
  cudaFree (cuda_other_vertex_out);
  cudaFree (cuda_other_vertex_in);
  cudaFree (cuda_other_vertex_mask_sum);
  cudaFree (cuda_other_vertex_mask);
  cudaFree (cuda_end_vertex_mask_sum);
  cudaFree (cuda_end_vertex_mask);
  cudaFree (cuda_last_vertex_mask);
  cudaFree (cuda_first_vertex_mask);
  cudaFree (reduced_weights);
  cudaFree (reduced_vertices_out);
  cudaFree (reduced_vertices_in);
  cudaFree (cuda_edge_mask_sum);
  cudaFree (cuda_edge_mask);

  cudaDeviceSynchronize();
  end = clock ();

  std::cout << blue << "CUDA infos: used memory " << (total_memory - free_memory)/pow(1024,3) << " Go / " << total_memory/pow(1024,3) << "Go" << reset;
  std::cout << "nb edges after cut on dataset = " << nb_edges << std::endl;
  std::cout << "nb vertices = " << nb_edges << std::endl;
  std::cout << "# first vertices = " << nb_first_vertices << std::endl;
  std::cout << "# other vertices = " << nb_other_vertices << std::endl;
  std::cout << "# hits in = " << nb_hitsID_in << std::endl;
  std::cout << "# last vertices = " << nb_last_vertices << std::endl;
  std::cout << "# end vertices = " << nb_end_vertices << std::endl;
  std::cout << "# vertices after kill on dead end vertices = " << nb_first_vertices - nb_not_dead_end_vertices << " / " << nb_first_vertices << std::endl;
  std::cout << "# second vertices = " << nb_second_vertices << std::endl;

  std::cout << magenta << "edge cuts: " << (long double)(end_cut-start_cut)/CLOCKS_PER_SEC << reset;
  std::cout << magenta << "scan: " << (long double)(end_scan-start_scan)/CLOCKS_PER_SEC << reset;
  std::cout << magenta << "stream reduction: " << (long double)(end_stream_reduction-start_stream_reduction)/CLOCKS_PER_SEC << reset;
  std::cout << magenta << "end vertex: " << (long double)(end_end_vertex - start_end_vertex)/CLOCKS_PER_SEC << reset;
  std::cout << magenta << "last vertex: " << (long double)(end_last_vertex - start_last_vertex)/CLOCKS_PER_SEC << reset;
  std::cout << magenta << "second vertex: " << (long double)(end_second_vertex - start_second_vertex)/CLOCKS_PER_SEC << reset;
  std::cout << magenta << "kill dead end connections: " << (long double)(end_dead_end_connections - start_dead_end_connections)/CLOCKS_PER_SEC << reset;
  //std::cout << "nb of edges = " << G.edges() << std::endl;
  //std::cout << "nb of edges as 1st component = " << G.edges() - reduced_edges.size() << std::endl;
  //std::cout << "nb of reduced edges = " << reduced_edges.size() << std::endl;
  std::cout << red << "incoming vertex cpu time : " << (long double)(end-start)/CLOCKS_PER_SEC << reset;
}
