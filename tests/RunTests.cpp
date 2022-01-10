#include <catch2/catch_session.hpp>

#include <Kokkos_Core.hpp>


// int main( int argc, char* argv[] ) {
//   Kokkos::initialize(argc,argv);

//   int result = Catch::Session().run( argc, argv );

//   Kokkos::finalize();
//   return result;
// }

int main( int argc, char* argv[] ) {
  Kokkos::initialize(argc,argv);

  Catch::Session session; // There must be exactly one instance

  int cores = 0; // Some user variable you want to be able to set

  // Build a new parser on top of Catch2's
  using namespace Catch::Clara;
  auto cli
    = session.cli() | Opt( cores, "kokkos-cores" ) ["--kokkos-cores"]("Number of cores to use with Kokkos.");

  // Now pass the new composite back to Catch2 so it uses that
  session.cli( cli );

  // Let Catch2 (using Clara) parse the command line
  int returnCode = session.applyCommandLine( argc, argv );
  if( returnCode != 0 ) // Indicates a command line error
      return returnCode;

  int result = session.run();

  Kokkos::finalize();
}