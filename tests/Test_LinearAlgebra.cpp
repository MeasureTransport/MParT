#include <catch2/catch_all.hpp>

#include "MParT/Utilities/LinearAlgebra.h"

using namespace mpart;
using namespace Catch;

TEST_CASE( "Testing Matrix Sum", "[Mat+Mat]" ) {

    Kokkos::View<double**, Kokkos::HostSpace> A("A", 2, 2);
    Kokkos::View<double** ,Kokkos::HostSpace> B("B", 2, 2);
    A(0,0) = 1.0; A(0,1) = 2.0;
    A(1,0) = 0.5; A(1,1) = 1.0;

    B(0,0) = 1.0; B(0,1) = 2.0;
    B(1,0) = 0.125; B(1,1) = 0.5;

    Kokkos::View<double**, Kokkos::HostSpace> C = A + B;
    for(unsigned int i=0; i<A.extent(0); ++i){
        for(unsigned int j=0; j<A.extent(1); ++j){
            CHECK(C(i,j) == Approx(A(i,j)+B(i,j)).epsilon(1e-15).margin(1e-15));
        }
    }

    A += B;
    for(unsigned int i=0; i<A.extent(0); ++i){
        for(unsigned int j=0; j<A.extent(1); ++j){
            CHECK(A(i,j) == Approx(C(i,j)).epsilon(1e-15).margin(1e-15));
        }
    }
}


TEST_CASE( "Testing Vector Sum", "[Vec+vec]" ) {

    Kokkos::View<double*, Kokkos::HostSpace> A("A", 2);
    Kokkos::View<double* ,Kokkos::HostSpace> B("B", 2);
    A(0) = 1.0;
    A(1) = 0.5;

    B(0) = 1.0;
    B(1) = 0.125;

    Kokkos::View<double*, Kokkos::HostSpace> C = A + B;
    for(unsigned int i=0; i<A.extent(0); ++i){
        CHECK(C(i) == Approx(A(i)+B(i)).epsilon(1e-15).margin(1e-15));
    }

    A += B;
    for(unsigned int i=0; i<A.extent(0); ++i){
        CHECK(A(i) == Approx(C(i)).epsilon(1e-15).margin(1e-15));
    }
}


TEST_CASE( "Testing Linear Mat-Mat product", "[LinearAlgebra_MatMat]" ) {

    Kokkos::View<double**, Kokkos::HostSpace> A("A", 2, 2);
    Kokkos::View<double** ,Kokkos::HostSpace> B("B", 2, 2);
    A(0,0) = 1.0; A(0,1) = 2.0;
    A(1,0) = 0.5; A(1,1) = 1.0;

    B(0,0) = 1.0; B(0,1) = 2.0;
    B(1,0) = 0.125; B(1,1) = 0.5;

    auto eigA = KokkosToMat(A);
    auto eigB = KokkosToMat(B);
    Eigen::MatrixXd trueC;
    StridedMatrix<double,Kokkos::HostSpace> C;

    SECTION("No Transpose") {
        C = A*B;
        trueC = eigA*eigB;

        REQUIRE(C.extent(0) == trueC.rows());
        REQUIRE(C.extent(1) == trueC.cols());

        for(unsigned int j=0; j<C.extent(1); ++j){
            for(unsigned int i=0; i<C.extent(0); ++i){
                CHECK(C(i,j) == Approx(trueC(i,j)).epsilon(1e-14).margin(1e-14));
            }
        }
    }

    SECTION("A Transpose") {
        C = transpose(A)*B;
        trueC = eigA.transpose()*eigB;

        REQUIRE(C.extent(0) == trueC.rows());
        REQUIRE(C.extent(1) == trueC.cols());

        for(unsigned int j=0; j<C.extent(1); ++j){
            for(unsigned int i=0; i<C.extent(0); ++i){
                CHECK(C(i,j) == Approx(trueC(i,j)).epsilon(1e-14).margin(1e-14));
            }
        }
    }

    SECTION("B Transpose") {
        C = A*transpose(B);
        trueC = eigA*eigB.transpose();

        REQUIRE(C.extent(0) == trueC.rows());
        REQUIRE(C.extent(1) == trueC.cols());

        for(unsigned int j=0; j<C.extent(1); ++j){
            for(unsigned int i=0; i<C.extent(0); ++i){
                CHECK(C(i,j) == Approx(trueC(i,j)).epsilon(1e-14).margin(1e-14));
            }
        }
    }

    SECTION("Both Transpose") {
        C = transpose(A)*transpose(B);
        trueC = eigA.transpose() * eigB.transpose();

        REQUIRE(C.extent(0) == trueC.rows());
        REQUIRE(C.extent(1) == trueC.cols());

        for(unsigned int j=0; j<C.extent(1); ++j){
            for(unsigned int i=0; i<C.extent(0); ++i){
                CHECK(C(i,j) == Approx(trueC(i,j)).epsilon(1e-14).margin(1e-14));
            }
        }
    }

    SECTION("Extra Transpose") {
        C = transpose(A)*transpose(transpose(B));
        trueC = eigA.transpose() * eigB;

        REQUIRE(C.extent(0) == trueC.rows());
        REQUIRE(C.extent(1) == trueC.cols());

        for(unsigned int j=0; j<C.extent(1); ++j){
            for(unsigned int i=0; i<C.extent(0); ++i){
                CHECK(C(i,j) == Approx(trueC(i,j)).epsilon(1e-14).margin(1e-14));
            }
        }
    }
}

TEST_CASE( "Testing LU Factorization", "LinearAlgebra_LU" ) {
    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> A("A", 3, 3);
    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> B("B", 3, 2);
    A(0,0) = 1.; A(0,1) = 2.; A(0,2) = 3.;
    A(1,0) = 4.; A(1,1) = 5.; A(1,2) = 6.;
    A(2,0) = 7.; A(2,1) = 8.; A(2,2) = 0.;

    B(0,0) = 1.; B(0,1) = 4.;
    B(1,0) = 2.; B(1,1) = 5.;
    B(2,0) = 3.; B(2,1) = 6.;

    auto constA = ToConstKokkos(A.data(), 3, 3);
    auto eigA = ConstKokkosToMat(constA);
    auto eigB = KokkosToMat(B);
    unsigned int nrows = eigA.rows();
    unsigned int ncols = eigA.cols();
    Eigen::MatrixXd eigX = eigA.lu().solve(eigB);

    SECTION("Compute LU") {
        PartialPivLU<Kokkos::HostSpace> Alu(constA);
    }
    SECTION("Solve LU inplace") {
        Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> C("C", 3, 2);
        Kokkos::deep_copy(C, B);
        PartialPivLU<Kokkos::HostSpace> Alu(constA);
        Alu.solveInPlace(C);
        for(unsigned int j=0; j<C.extent(1); ++j){
            for(unsigned int i=0; i<C.extent(0); ++i){
                CHECK(C(i,j) == Approx(eigX(i,j)).epsilon(1e-14).margin(1e-14));
            }
        }
    }
    SECTION("Solve LU out of place") {
        PartialPivLU<Kokkos::HostSpace> Alu(constA);
        auto C = Alu.solve(B);
        for(unsigned int j=0; j<C.extent(1); ++j){
            for(unsigned int i=0; i<C.extent(0); ++i){
                CHECK(C(i,j) == Approx(eigX(i,j)).epsilon(1e-14).margin(1e-14));
            }
        }
    }
    SECTION("Compute Determinant") {
        PartialPivLU<Kokkos::HostSpace> Alu(constA);
        CHECK(Alu.determinant() == Approx(eigA.determinant()).epsilon(1e-14).margin(1e-14));
    }
}

TEST_CASE( "Testing Cholesky Factorization", "LinearAlgebra_Cholesky" ) {
    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> A("A", 3, 3);
    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> B("B", 3, 2);
    A(0,0) = 36.; A(0,1) = 18.; A(0,2) = 6.;
    A(1,0) = 18.; A(1,1) = 34.; A(1,2) = 13.;
    A(2,0) = 6.;  A(2,1) = 13.; A(2,2) = 21.;

    B(0,0) = 1.; B(0,1) = 4.;
    B(1,0) = 2.; B(1,1) = 5.;
    B(2,0) = 3.; B(2,1) = 6.;

    auto constA = ToConstKokkos(A.data(), 3, 3);
    auto eigA = ConstKokkosToMat(constA);
    auto eigB = KokkosToMat(B);
    unsigned int nrows = eigA.rows();
    unsigned int ncols = eigA.cols();
    Eigen::MatrixXd eigX = eigA.llt().solve(eigB);
    Eigen::MatrixXd eigY = eigA.llt().matrixL().solve(eigB);

    SECTION("Compute Cholesky") {
        Cholesky<Kokkos::HostSpace> Achol(constA);
    }
    SECTION("Solve Cholesky inplace") {
        Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> C("C", 3, 2);
        Kokkos::deep_copy(C, B);
        Cholesky<Kokkos::HostSpace> Achol(constA);
        Achol.solveInPlace(C);
        for(unsigned int j=0; j<C.extent(1); ++j){
            for(unsigned int i=0; i<C.extent(0); ++i){
                CHECK(C(i,j) == Approx(eigX(i,j)).epsilon(1e-14).margin(1e-14));
            }
        }
    }
    SECTION("Solve Cholesky L inplace") {
        Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> C("C", 3, 2);
        Kokkos::deep_copy(C, B);
        Cholesky<Kokkos::HostSpace> Achol(constA);
        Achol.solveLInPlace(C);
        for(unsigned int j=0; j<C.extent(1); ++j){
            for(unsigned int i=0; i<C.extent(0); ++i){
                CHECK(C(i,j) == Approx(eigY(i,j)).epsilon(1e-14).margin(1e-14));
            }
        }
    }
    SECTION("Solve Cholesky out of place") {
        Cholesky<Kokkos::HostSpace> Achol(constA);
        auto C = Achol.solve(B);
        for(unsigned int j=0; j<C.extent(1); ++j){
            for(unsigned int i=0; i<C.extent(0); ++i){
                CHECK(C(i,j) == Approx(eigX(i,j)).epsilon(1e-14).margin(1e-14));
            }
        }
    }
    SECTION("Compute Determinant") {
        Cholesky<Kokkos::HostSpace> Achol(constA);
        CHECK(Achol.determinant() == Approx(eigA.determinant()).epsilon(1e-14).margin(1e-14));
    }
}

#if defined(MPART_ENABLE_GPU)

TEST_CASE( "Testing Linear Mat-Mat product on Device", "[LinearAlgebra_MatMatDevice]" ) {

    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> A("A", 2, 2);
    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> B("B", 2, 2);
    A(0,0) = 1.0; A(0,1) = 2.0;
    A(1,0) = 0.5; A(1,1) = 1.0;

    B(0,0) = 1.0; B(0,1) = 2.0;
    B(1,0) = 0.125; B(1,1) = 0.5;

    auto dA = ToDevice<mpart::DeviceSpace>(A);
    auto dB = ToDevice<mpart::DeviceSpace>(B);

    auto eigA = KokkosToMat(A);
    auto eigB = KokkosToMat(B);
    Eigen::MatrixXd trueC;
    StridedMatrix<double,Kokkos::HostSpace> C;
    StridedMatrix<double,mpart::DeviceSpace> dC;

    SECTION("No Transpose") {
        dC = dA*dB;
        C = ToHost(dC);

        trueC = eigA*eigB;

        REQUIRE(C.extent(0) == trueC.rows());
        REQUIRE(C.extent(1) == trueC.cols());

        for(unsigned int j=0; j<C.extent(1); ++j){
            for(unsigned int i=0; i<C.extent(0); ++i){
                CHECK(C(i,j) == Approx(trueC(i,j)).epsilon(1e-14).margin(1e-14));
            }
        }
    }

    SECTION("A Transpose") {
        dC = transpose(dA)*dB;
        C = ToHost(dC);
        trueC = eigA.transpose()*eigB;

        REQUIRE(C.extent(0) == trueC.rows());
        REQUIRE(C.extent(1) == trueC.cols());

        for(unsigned int j=0; j<C.extent(1); ++j){
            for(unsigned int i=0; i<C.extent(0); ++i){
                CHECK(C(i,j) == Approx(trueC(i,j)).epsilon(1e-14).margin(1e-14));
            }
        }
    }

    SECTION("B Transpose") {
        dC = dA*transpose(dB);
        C = ToHost(dC);
        trueC = eigA*eigB.transpose();

        REQUIRE(C.extent(0) == trueC.rows());
        REQUIRE(C.extent(1) == trueC.cols());

        for(unsigned int j=0; j<C.extent(1); ++j){
            for(unsigned int i=0; i<C.extent(0); ++i){
                CHECK(C(i,j) == Approx(trueC(i,j)).epsilon(1e-14).margin(1e-14));
            }
        }
    }

    SECTION("Both Transpose") {
        dC = transpose(dA)*transpose(dB);
        C = ToHost(dC);
        trueC = eigA.transpose() * eigB.transpose();

        REQUIRE(C.extent(0) == trueC.rows());
        REQUIRE(C.extent(1) == trueC.cols());

        for(unsigned int j=0; j<C.extent(1); ++j){
            for(unsigned int i=0; i<C.extent(0); ++i){
                CHECK(C(i,j) == Approx(trueC(i,j)).epsilon(1e-14).margin(1e-14));
            }
        }
    }

    SECTION("Extra Transpose") {
        dC = transpose(dA)*transpose(transpose(dB));
        C = ToHost(dC);
        trueC = eigA.transpose() * eigB;

        REQUIRE(C.extent(0) == trueC.rows());
        REQUIRE(C.extent(1) == trueC.cols());

        for(unsigned int j=0; j<C.extent(1); ++j){
            for(unsigned int i=0; i<C.extent(0); ++i){
                CHECK(C(i,j) == Approx(trueC(i,j)).epsilon(1e-14).margin(1e-14));
            }
        }
    }
}

TEST_CASE( "Testing LU Factorization on Device", "LinearAlgebra_LUDevice" ) {
    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> A("A", 3, 3);
    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> B("B", 3, 2);
    A(0,0) = 1.; A(0,1) = 2.; A(0,2) = 3.;
    A(1,0) = 4.; A(1,1) = 5.; A(1,2) = 6.;
    A(2,0) = 7.; A(2,1) = 8.; A(2,2) = 0.;

    B(0,0) = 1.; B(0,1) = 4.;
    B(1,0) = 2.; B(1,1) = 5.;
    B(2,0) = 3.; B(2,1) = 6.;

    auto constA = ToConstKokkos(A.data(), 3, 3);
    auto constA_d = ToDevice<mpart::DeviceSpace>(constA);
    auto B_d = ToDevice<mpart::DeviceSpace>(B);
    auto eigA = ConstKokkosToMat(constA);
    auto eigB = KokkosToMat(B);
    unsigned int nrows = eigA.rows();
    unsigned int ncols = eigA.cols();
    Eigen::MatrixXd eigX = eigA.lu().solve(eigB);

    SECTION("Compute LU") {
        PartialPivLU<mpart::DeviceSpace> Alu_d(constA_d);
    }
    SECTION("Solve LU inplace") {
        Kokkos::View<double**, Kokkos::LayoutLeft, mpart::DeviceSpace> C_d("C", 3, 2);
        Kokkos::deep_copy(C_d, B_d);
        PartialPivLU<mpart::DeviceSpace> Alu_d(constA_d);
        Alu_d.solveInPlace(C_d);
        auto C_h = ToHost(C_d);
        for(unsigned int j=0; j<C_h.extent(1); ++j){
            for(unsigned int i=0; i<C_h.extent(0); ++i){
                CHECK(C_h(i,j) == Approx(eigX(i,j)).epsilon(1e-14).margin(1e-14));
            }
        }
    }
    SECTION("Solve LU out of place") {
        PartialPivLU<mpart::DeviceSpace> Alu_d(constA_d);
        auto C_d = Alu.solve(B_d);
        auto C_h = ToHost(C_d);
        for(unsigned int j=0; j<C_h.extent(1); ++j){
            for(unsigned int i=0; i<C_h.extent(0); ++i){
                CHECK(C_h(i,j) == Approx(eigX(i,j)).epsilon(1e-14).margin(1e-14));
            }
        }
    }
    SECTION("Compute Determinant") {
        PartialPivLU<Kokkos::HostSpace> Alu_d(constA_d);
        CHECK(Alu_d.determinant() == Approx(eigA.determinant()).epsilon(1e-14).margin(1e-14));
    }
}

TEST_CASE( "Testing Cholesky Factorization", "LinearAlgebra_Cholesky" ) {
    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> A("A", 3, 3);
    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> B_h("B", 3, 2);
    A(0,0) = 36.; A(0,1) = 18.; A(0,2) = 6.;
    A(1,0) = 18.; A(1,1) = 34.; A(1,2) = 13.;
    A(2,0) = 6.;  A(2,1) = 13.; A(2,2) = 21.;

    B_h(0,0) = 1.; B_h(0,1) = 4.;
    B_h(1,0) = 2.; B_h(1,1) = 5.;
    B_h(2,0) = 3.; B_h(2,1) = 6.;

    auto constA_h = ToConstKokkos(A.data(), 3, 3);
    auto constA_d = ToDevice<mpart::DeviceSpace>(constA_h);
    auto B_d = ToDevice<mpart::DeviceSpace>(B);
    auto eigA = ConstKokkosToMat(constA_h);
    auto eigB = KokkosToMat(B);
    Eigen::MatrixXd eigX = eigA.llt().solve(eigB);
    Eigen::MatrixXd eigY = eigA.llt().matrixL().solve(eigB);

    SECTION("Compute Cholesky") {
        Cholesky<Kokkos::HostSpace> Achol_d(constA_d);
    }
    SECTION("Solve Cholesky inplace") {
        Kokkos::View<double**, Kokkos::LayoutLeft, mpart::DeviceSpace> C_d("C", 3, 2);
        Kokkos::deep_copy(C_d, B_d);
        Cholesky<Kokkos::HostSpace> Achol_d(constA_d);
        Achol_d.solveInPlace(C_d);
        auto C_h = ToHost(C_d);
        for(unsigned int j=0; j<C_h.extent(1); ++j){
            for(unsigned int i=0; i<C_h.extent(0); ++i){
                CHECK(C_h(i,j) == Approx(eigX(i,j)).epsilon(1e-14).margin(1e-14));
            }
        }
    }
    SECTION("Solve Cholesky L inplace") {
        Kokkos::View<double**, Kokkos::LayoutLeft, mpart::DeviceSpace> C_d("C", 3, 2);
        Kokkos::deep_copy(C_d, B_d);
        Cholesky<Kokkos::HostSpace> Achol_d(constA_d);
        Achol_d.solveLInPlace(C_d);
        auto C_h = ToHost(C_d);
        for(unsigned int j=0; j<C_h.extent(1); ++j){
            for(unsigned int i=0; i<C_h.extent(0); ++i){
                CHECK(C_h(i,j) == Approx(eigY(i,j)).epsilon(1e-14).margin(1e-14));
            }
        }
    }
    SECTION("Solve Cholesky out of place") {
        Cholesky<Kokkos::HostSpace> Achol_d(constA_d);
        auto C_d = Achol.solve(B_d);
        auto C_h = ToHost(C_d);
        for(unsigned int j=0; j<C_h.extent(1); ++j){
            for(unsigned int i=0; i<C_h.extent(0); ++i){
                CHECK(C_h(i,j) == Approx(eigX(i,j)).epsilon(1e-14).margin(1e-14));
            }
        }
    }
    SECTION("Compute Determinant") {
        Cholesky<Kokkos::HostSpace> Achol_d(constA_d);
        CHECK(Achol_d.determinant() == Approx(eigA.determinant()).epsilon(1e-14).margin(1e-14));
    }
}
#endif