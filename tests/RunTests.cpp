#include <catch2/catch_session.hpp>

#include "MParT/Initialization.h"

int main( int argc, char* argv[] ) {
  mpart::Initialize(argc,argv);

  std::cout << "Kokkos Concurrency = " << Kokkos::DefaultExecutionSpace::concurrency() << std::endl;

  Catch::Session session; // There must be exactly one instance

  int cores = 0; // Some user variable you want to be able to set

  // Build a new parser on top of Catch2's
  using namespace Catch::Clara;
  auto cli
    = session.cli() | Opt( cores, "kokkos-thread" ) ["--kokkos-threads"]("Number of cores to use with Kokkos.");

  // Now pass the new composite back to Catch2 so it uses that
  session.cli( cli );

  // Let Catch2 (using Clara) parse the command line
  int returnCode = session.applyCommandLine( argc, argv );
  if( returnCode != 0 ) // Indicates a command line error
      return returnCode;

  session.run();

}