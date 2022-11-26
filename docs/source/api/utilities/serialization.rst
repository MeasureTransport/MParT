====================
Serialization
====================
MParT supports the ability to serialize/archive objects when installed with the :code:`MPART_ARCHIVE` cmake configuration option, powered by the cereal C++ serialization library. One example of using this to save would be to do


.. tab-set::

    .. tab-item:: C++

        .. code-block:: c++

            #include <MParT/Utilities/Serialization.h>
            using namespace mpart;

            void SerializeComponent(int inputDim, int maxOrder, std::string filename) {
                // Create MapOptions
                MapOptions options;
                options.basisType = BasisTypes::ProbabilistHermite;
                options.basisNorm = false;

                // Create FixedMultiIndexSet and map component
                FixedMultiIndexSet<Kokkos::HostSpace> mset (inputDim, maxOrder);
                auto comp = MapFactory::CreateComponent(mset, options);
                auto coeffs = comp->Coeffs();
                for (int i = 0; i < coeffs.extent(0); i++)
                    coeffs(i) = 0.5*(i+1);

                // Serialize objects
                std::ofstream os(filename);
                cereal::BinaryOutputArchive archive(os);
                archive(options, mset, comp->Coeffs());
            }

            ConditionalMapBase<Kokkos::HostSpace> DeserializeComponent(std::string filename) {
                std::ifstream is(filename);
                cereal::BinaryInputArchive archive(is);
                MapOptions options;
                FixedMultiIndexSet<Kokkos::HostSpace> mset(1,1); // No default constructor
                Kokkos::View<double*, Kokkos::HostSpace> coeffs;
                archive(options, mset, coeffs);
                auto comp = MapFactory::CreateComponent(mset, options);
                comp->WrapCoeffs(coeffs);
                return comp;
            }

            int main() {
                int inputDim = 5;
                int maxOrder = 3;
                std::string filename = "component.mt";
                SerializeComponent(inputDim, maxOrder, filename);
                auto comp = DeserializeComponent(filename);
            }


    .. tab-item:: Python

        In Python, you can only serialize one object per file.

        .. code-block:: python

            import mpart as mt

            def SerializeComponent(inputDim, maxOrder):
                # Create MapOptions
                options = mt.MapOptions()
                options.basisType = mt.BasisTypes.ProbabilistHermite
                options.basisNorm = False;

                # Create FixedMultiIndexSet and map component
                multis = np.array([[0],[1]])
                mset= mt.MultiIndexSet(multis)
                fixed_mset = mset.fix(True)

                component = mt.CreateComponent(fixed_mset, options)
                coeffs = component.CoeffMap()
                for i in range(len(coeffs)):
                    coeffs[i] = 0.5*(i+1)

                # Serialize objects
                fixed_mset.Serialize("fmset.mt")
                options.Serialize("opts.mt")
                component.Serialize("comp.mt")

            def DeserializeComponent():
                # Deserialize the FixedMultiIndexSet
                # Note that we need to construct the object before calling Deserialize
                fixed_mset = mt.MultiIndexSet(np.array([[0]])).fix(True)
                fixed_mset.Deserialize("fmset.mt")

                # Deserialize the MapOptions
                options = MapOptions()
                options.Deserialize("opts.mt")

                # Deserialize the Map Coefficients and construct the component
                inputDim, outputDim, coeffs = mt.DeserializeMap("comp.mt")
                component = mt.CreateComponent(fixed_mset, options)
                component.SetCoeffs(coeffs)
