/***************************************************************
 * Tracking project library - L2IT
 * Trace reconstruction in LHC
 * copyright © 2023,2024 COLLARD Christophe
 * copyright © 2023,2024 Centre National de la Recherche Scientifique
 * copyright © 2023,2024 Laboratoire des 2 Infinis de Toulouse (L2IT)
 ***************************************************************/

#ifndef __cplusplus
#error Must use C++ for MPI receive
#endif

#include <mpi.h>
#include <module_map_triplet>


//-----------------------------------------------------------------------------------------------------
template <class T>
void receive (module_doublet<T>& ModuleDoublet, int from_thread, int tag, MPI::Datatype& MPI_data_type)
//-----------------------------------------------------------------------------------------------------
{
  MPI::COMM_WORLD.Recv (&ModuleDoublet.z0_min(), 1, MPI_data_type, from_thread, tag);
  MPI::COMM_WORLD.Recv (&ModuleDoublet.z0_max(), 1, MPI_data_type, from_thread, tag);
  MPI::COMM_WORLD.Recv (&ModuleDoublet.dphi_min(), 1, MPI_data_type, from_thread, tag);
  MPI::COMM_WORLD.Recv (&ModuleDoublet.dphi_max(), 1, MPI_data_type, from_thread, tag);
  MPI::COMM_WORLD.Recv (&ModuleDoublet.phi_slope_min(), 1, MPI_data_type, from_thread, tag);
  MPI::COMM_WORLD.Recv (&ModuleDoublet.phi_slope_max(), 1, MPI_data_type, from_thread, tag);
  MPI::COMM_WORLD.Recv (&ModuleDoublet.deta_min(), 1, MPI_data_type, from_thread, tag);
  MPI::COMM_WORLD.Recv (&ModuleDoublet.deta_max(), 1, MPI_data_type, from_thread, tag);
}

//-----------------------------------------------------------------------
template <class T>
void receive (module_triplet<T>& ModuleTriplet, int from_thread, int tag)
//-----------------------------------------------------------------------
{
  MPI::COMM_WORLD.Recv (&ModuleTriplet.occurence(), 1, MPI::UNSIGNED, from_thread, tag);

  MPI::Datatype cut_type = MPI::BYTE.Create_hvector (1, sizeof(T), 1);
  cut_type.Commit();
  MPI::COMM_WORLD.Recv (&ModuleTriplet.diff_dydx_min(), 1, cut_type, from_thread, tag);
  MPI::COMM_WORLD.Recv (&ModuleTriplet.diff_dydx_max(), 1, cut_type, from_thread, tag);
  MPI::COMM_WORLD.Recv (&ModuleTriplet.diff_dzdr_min(), 1, cut_type, from_thread, tag);
  MPI::COMM_WORLD.Recv (&ModuleTriplet.diff_dzdr_max(), 1, cut_type, from_thread, tag);
  cut_type.Free();
  exit(0); // complete this with module duet MPI_send
}

//-----------------------------------------------------------------------------------------------------
template <class T>
void receive (module_triplet<T>& ModuleTriplet, int from_thread, int tag, MPI::Datatype& MPI_data_type)
//-----------------------------------------------------------------------------------------------------
{
  MPI::COMM_WORLD.Recv (&ModuleTriplet.occurence(), 1, MPI::UNSIGNED, from_thread, tag);
  MPI::COMM_WORLD.Recv (&ModuleTriplet.diff_dydx_min(), 1, MPI_data_type, from_thread, tag);
  MPI::COMM_WORLD.Recv (&ModuleTriplet.diff_dydx_max(), 1, MPI_data_type, from_thread, tag);
  MPI::COMM_WORLD.Recv (&ModuleTriplet.diff_dzdr_min(), 1, MPI_data_type, from_thread, tag);
  MPI::COMM_WORLD.Recv (&ModuleTriplet.diff_dzdr_max(), 1, MPI_data_type, from_thread, tag);

  receive (ModuleTriplet.modules12(), from_thread, tag, MPI_data_type);
  receive (ModuleTriplet.modules23(), from_thread, tag, MPI_data_type);
}

//------------------------------------------------------------------------------
template <class T>
void receive (module_map_triplet<T>& ModuleMapTriplet, int from_thread, int tag)
//------------------------------------------------------------------------------
{
  int nb_threads = MPI::COMM_WORLD.Get_size();
  int my_rank = MPI::COMM_WORLD.Get_rank();
  assert (from_thread>=0 && from_thread<nb_threads);

  // receivesend module triplets
  int size;
  MPI::COMM_WORLD.Recv (&size, 1, MPI::INT, from_thread, tag);
  MPI::Datatype MPI_data_type = MPI::BYTE.Create_hvector (1, sizeof(T), 1);
  MPI_data_type.Commit();

  for (int i=0; i< size; i++) {
    std::vector<uint64_t> triplet(3);
    MPI::COMM_WORLD.Recv (&triplet[0], 3, MPI_UNSIGNED_LONG, from_thread, tag);
    module_triplet<T> ModuleTriplet;
    receive (ModuleTriplet, from_thread, tag, MPI_data_type);
    ModuleMapTriplet.map_triplet().insert (std::pair<std::vector<uint64_t>, module_triplet<T>> (triplet, ModuleTriplet));
  }

  MPI_data_type.Free();

  // receive module doublets
  MPI::COMM_WORLD.Recv (&size, 1, MPI::INT, from_thread, tag);
  MPI_data_type = MPI::BYTE.Create_hvector (1, sizeof(T), 1);
  MPI_data_type.Commit();

  for (int i=0; i< size; i++) {
    std::vector<uint64_t> doublet(2);
    MPI::COMM_WORLD.Recv (&doublet[0], 2, MPI_UNSIGNED_LONG, from_thread, tag);
    module_doublet<T> ModuleDoublet;
    receive (ModuleDoublet, from_thread, tag, MPI_data_type);
    ModuleMapTriplet.map_doublet().insert (std::pair<std::vector<uint64_t>, module_doublet<T>> (doublet, ModuleDoublet));
  }

  MPI_data_type.Free();
}
