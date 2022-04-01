#include <Kokkos_Core.hpp>

#include <iostream>

int main( int argc, char* argv[] ) {
    Kokkos::initialize(argc,argv);

    Kokkos::print_configuration(std::cout);

    Kokkos::finalize();
    return 0;
}