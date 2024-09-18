/***************************************************************
 * Tracking project library - L2IT
 * Trace reconstruction in LHC
 * copyright © 2022,2023 ROUGIER Charline
 * copyright © 2022,2023 COLLARD Christophe
 * copyright © 2022,2023 Centre National de la Recherche Scientifique
 * copyright © 2022,2023 Laboratoire des 2 Infinis de Toulouse (L2IT)
 ***************************************************************/

#ifndef __cplusplus
#error Must use C++ for the type vectex true
#endif

#include <vertex>

template <class T> class graph_true;

//=====================================================
template <class T> class vertex_true : public vertex<T>
//=====================================================
{
  friend void graph_true<T>::save (const std::string&, const std::string&);

  protected:
    T _pt_particle;
    T _eta;

  public:
    vertex_true () : vertex<T>() {_pt_particle = _eta = 0;}
    T& pT () {return _pt_particle;}
    T& eta () {return _eta;}
};
