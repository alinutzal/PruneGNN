/***************************************************************
 * Tracking project library - L2IT
 * Trace reconstruction in LHC
 * copyright © 2023,2024 COLLARD Christophe
 * copyright © 2023,2024 Centre National de la Recherche Scientifique
 * copyright © 2023,2024 Laboratoire des 2 Infinis de Toulouse (L2IT)
 ***************************************************************/

#ifndef __cplusplus
#error Must use C++ for MPI send
#endif

#include <mpi.h>
#include <module_triplet>
#include <module_map_triplet>


//------------------------------------------------------------------------------------------------------
template <class T>
void send (const module_doublet<T>& ModuleDoublet, int to_thread, int tag, MPI::Datatype& MPI_data_type)
//------------------------------------------------------------------------------------------------------
{
  MPI::COMM_WORLD.Send (&ModuleDoublet.z0_min(), 1, MPI_data_type, to_thread, tag);
  MPI::COMM_WORLD.Send (&ModuleDoublet.z0_max(), 1, MPI_data_type, to_thread, tag);
  MPI::COMM_WORLD.Send (&ModuleDoublet.dphi_min(), 1, MPI_data_type, to_thread, tag);
  MPI::COMM_WORLD.Send (&ModuleDoublet.dphi_max(), 1, MPI_data_type, to_thread, tag);
  MPI::COMM_WORLD.Send (&ModuleDoublet.phi_slope_min(), 1, MPI_data_type, to_thread, tag);
  MPI::COMM_WORLD.Send (&ModuleDoublet.phi_slope_max(), 1, MPI_data_type, to_thread, tag);
  MPI::COMM_WORLD.Send (&ModuleDoublet.deta_min(), 1, MPI_data_type, to_thread, tag);
  MPI::COMM_WORLD.Send (&ModuleDoublet.deta_max(), 1, MPI_data_type, to_thread, tag);
}

//------------------------------------------------------------------------
template <class T>
void send (const module_triplet<T>& ModuleTriplet, int to_thread, int tag)
//------------------------------------------------------------------------
{
  MPI::COMM_WORLD.Send (&ModuleTriplet.occurence(), 1, MPI::UNSIGNED, to_thread, tag);

  MPI::Datatype cut_type = MPI::BYTE.Create_hvector (1, sizeof(T), 1);
  cut_type.Commit();
  MPI::COMM_WORLD.Send (&ModuleTriplet.diff_dydx_min(), 1, cut_type, to_thread, tag);
  MPI::COMM_WORLD.Send (&ModuleTriplet.diff_dydx_max(), 1, cut_type, to_thread, tag);
  MPI::COMM_WORLD.Send (&ModuleTriplet.diff_dzdr_min(), 1, cut_type, to_thread, tag);
  MPI::COMM_WORLD.Send (&ModuleTriplet.diff_dzdr_max(), 1, cut_type, to_thread, tag);
  cut_type.Free();

  send (ModuleTriplet.modules12(), to_thread, tag);
  send (ModuleTriplet.modules23(), to_thread, tag);
}

//------------------------------------------------------------------------------------------------------
template <class T>
void send (const module_triplet<T>& ModuleTriplet, int to_thread, int tag, MPI::Datatype& MPI_data_type)
//------------------------------------------------------------------------------------------------------
{
  MPI::COMM_WORLD.Send (&ModuleTriplet.occurence(), 1, MPI::UNSIGNED, to_thread, tag);
  MPI::COMM_WORLD.Send (&ModuleTriplet.diff_dydx_min(), 1, MPI_data_type, to_thread, tag);
  MPI::COMM_WORLD.Send (&ModuleTriplet.diff_dydx_max(), 1, MPI_data_type, to_thread, tag);
  MPI::COMM_WORLD.Send (&ModuleTriplet.diff_dzdr_min(), 1, MPI_data_type, to_thread, tag);
  MPI::COMM_WORLD.Send (&ModuleTriplet.diff_dzdr_max(), 1, MPI_data_type, to_thread, tag);

  send (ModuleTriplet.modules12(), to_thread, tag, MPI_data_type);
  send (ModuleTriplet.modules23(), to_thread, tag, MPI_data_type);
}

//-------------------------------------------------------------------------------
template <class T>
void send (const module_map_triplet<T>& ModuleMapTriplet, int to_thread, int tag)
//-------------------------------------------------------------------------------
{
  int nb_threads = MPI::COMM_WORLD.Get_size();
  int my_rank = MPI::COMM_WORLD.Get_rank();
  assert (to_thread>=0 && to_thread<nb_threads);

  // send module map triplets
  int size = ModuleMapTriplet.size();
  MPI::COMM_WORLD.Send (&size, 1, MPI::INT, to_thread, tag);

  MPI::Datatype MPI_data_type = MPI::BYTE.Create_hvector (1, sizeof(T), 1);
  MPI_data_type.Commit();

  for (const std::pair <const std::vector<uint64_t>, module_triplet<T> > MMentry : ModuleMapTriplet.map_triplet()) {
    MPI::COMM_WORLD.Send (&MMentry.first[0], 3, MPI::UNSIGNED_LONG, to_thread, tag);
    send (MMentry.second, to_thread, tag, MPI_data_type);
  }

  MPI_data_type.Free();

  // sending module doublets
  size = ModuleMapTriplet.map_doublet().size();
  MPI::COMM_WORLD.Send (&size, 1, MPI::INT, to_thread, tag);

  MPI_data_type = MPI::BYTE.Create_hvector (1, sizeof(T), 1);
  MPI_data_type.Commit();

  for (const std::pair <const std::vector<uint64_t>, module_doublet<T>> MMentry : ModuleMapTriplet.map_doublet()) {
    MPI::COMM_WORLD.Send (&MMentry.first[0], 2, MPI::UNSIGNED_LONG, to_thread, tag);
    send (MMentry.second, to_thread, tag, MPI_data_type);
  }

  MPI_data_type.Free();
}
