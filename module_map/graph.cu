/***************************************************************
 * Tracking project library - L2IT
 * Trace reconstruction in LHC
 * copyright © 2024 COLLARD Christophe
 * copyright © 2024 Centre National de la Recherche Scientifique
 * copyright © 2024 Laboratoire des 2 Infinis de Toulouse (L2IT)
 ***************************************************************/

#include "graph.cuh"

//-----------------------------------------------------------------------
__global__ void printdata (int size, int* overtex)
//-----------------------------------------------------------------------
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) return;

//  first_vertex[overtex[i]] = 0;
  printf ("i = %d / vertex = %d \n", i, overtex[i]);
}


//=====Private methods for graph=============================================

//=====Public methods for graph==============================================

//--------------------------------------------------------------------------------------------------------------------------------
template <class T>
CUDA_graph<T>::CUDA_graph (const std::string& events_dir, const std::string& vertices_filename, const std::string& edges_filename)
//--------------------------------------------------------------------------------------------------------------------------------
{
  _vertices_size = _edges_size = 0;
  char delimiter = ',';

  // read nodes
  std::string path_ReadNodes = events_dir + vertices_filename;
  std::ifstream file_vertices (path_ReadNodes);
  if (file_vertices.fail()) throw std::invalid_argument ("Cannot open file " + path_ReadNodes);

  std::string csv_content;
  file_vertices >> csv_content;  // delete first line = comments


  std::string v;
  std::vector<int> vertex;
  file_vertices >> csv_content;  // read 1st non comment line
  for (; !file_vertices.eof(); _vertices_size++) {
    std::stringstream ss (csv_content);
    getline (ss, v, delimiter);
    vertex.push_back (std::stoi(v));
    file_vertices >> csv_content;  // delete first line = comments
  }

  file_vertices.close ();

  // read edges
  std::string path_ReadGraphs = events_dir + edges_filename;
  std::ifstream file_edges (path_ReadGraphs);
  if (file_edges.fail()) throw std::invalid_argument ("Cannot open file " + path_ReadGraphs);

  file_edges >> csv_content;  // delete first line = comments

  std::string v1, v2, w;
  std::vector<int> input_vertex, output_vertex;
  std::vector<T> weight;
  file_edges >> csv_content;  // delete first line = comments

  for (; !file_edges.eof(); _edges_size++) {
    std::stringstream ss (csv_content);
    getline (ss, v1, delimiter);
    getline (ss, v2, delimiter);
    getline (ss, w, delimiter);
    input_vertex.push_back (std::stoi(v1));
    output_vertex.push_back (std::stoi(v2));
    weight.push_back (std::stod(w));
    file_edges >> csv_content;  // read first non comment line
  }

  file_edges.close();

  _start = clock();
  // sort on output vertices
  std::multimap <int, int> sorted_output_vertices;
  for (int i=0; i<_edges_size; i++)
    sorted_output_vertices.insert (std::pair<int,int> (output_vertex[i], i));

  std::vector<int> sorted_output_vertices_edges;
  for (std::pair<int,int> vertex :sorted_output_vertices)
    sorted_output_vertices_edges.push_back(vertex.second);

//for (int i=0; i<vertex.size(); i++)
//    std::cout << "vertex = " << vertex[i] << std::endl;
//for (int i=0; i<_edges_size; i++)
//    std::cout << "edges = " << input_vertex[i] << " " << output_vertex[i] << std::endl;
std::cout << "nb vertices = " << _vertices_size << std::endl;
std::cout << "nb edges = " << _edges_size << std::endl;
  // copy data to device
  cudaMalloc (&_vertex, _vertices_size * sizeof(int));
  cudaMalloc (&_ivertex, _edges_size * sizeof(int));
  cudaMalloc (&_overtex, _edges_size * sizeof(int));
  cudaMalloc (&_sorted_overtex, _edges_size * sizeof(int));
  cudaMalloc (&_weight, _edges_size * sizeof(int));
  cudaMemcpy (_vertex, vertex.data(), _vertices_size * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy (_ivertex, input_vertex.data(), _edges_size * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy (_overtex, output_vertex.data(), _edges_size * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy (_sorted_overtex, sorted_output_vertices_edges.data(), _edges_size * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy (_weight, weight.data(), _edges_size * sizeof(T), cudaMemcpyHostToDevice);
}

//---------------------------
template <class T>
CUDA_graph<T>::~CUDA_graph ()
//---------------------------
{
  cudaFree (_weight);
  cudaFree (_sorted_overtex);
  cudaFree (_overtex);
  cudaFree (_ivertex);
  cudaFree (_vertex);
  _end = clock();

  std::cout << blue << "1 event walk through cpu time : " << (long double)(_end-_start)/CLOCKS_PER_SEC << std::endl;
  std::cout << "end read file" << std::endl;
  int time = (long double)(_end-_start)/CLOCKS_PER_SEC;
  int min = time / 60;
  int sec = time - 60 * min;
  std::cout << green << "1 event walk through cpu time : " << min << "'" << sec << "\"" << reset;
}
