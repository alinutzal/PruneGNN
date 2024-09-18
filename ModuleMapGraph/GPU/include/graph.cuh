/***************************************************************
 * Tracking project library - L2IT
 * Trace reconstruction in LHC
 * copyright © 2024 COLLARD Christophe
 * copyright © 2024 Centre National de la Recherche Scientifique
 * copyright © 2024 Laboratoire des 2 Infinis de Toulouse (L2IT)
 ***************************************************************/

#ifndef __cplusplus
#error Must use C++ for the type CUDA graph
#endif

#include <string>
#include <iostream>
#include <boost/regex.hpp>
#include <colors>

//=================================
template <class T> class CUDA_graph
//=================================
{
  private:
    int _vertices_size, _edges_size;
    int *_vertex;
    int *_ivertex, *_overtex, *_sorted_overtex;
    T *_weight;
    clock_t _start, _end;

  public:

    CUDA_graph (const std::string&, const std::string&, const std::string&);
    ~CUDA_graph ();

    __device__ inline int* ivertex () {return _ivertex;}
    __device__ inline int* overtex () {return _overtex;}

    inline int vertices () const {return _vertices_size;}
    inline int edges () const {return _edges_size;}
    inline int* vertices_in () const {return _ivertex;}
    inline int* vertices_out () const {return _overtex;}
    inline T* weights () const {return _weight;}
    inline int* hit_id () const {return _vertex;}
};
