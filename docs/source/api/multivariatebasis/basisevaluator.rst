=================
Basis Evaluators
=================

.. doxygenenum:: mpart::BasisHomogeneity

.. doxygenclass:: mpart::BasisEvaluator
    :members:
    :undoc-members:

.. doxygenclass:: mpart::BasisEvaluator< BasisHomogeneity::Homogeneous, BasisEvaluatorType >

.. doxygenclass:: mpart::BasisEvaluator< BasisHomogeneity::OffdiagHomogeneous, Kokkos::pair< OffdiagEvaluatorType, DiagEvaluatorType > >

.. doxygenclass:: mpart::BasisEvaluator< BasisHomogeneity::Heterogeneous, std::vector< std::shared_ptr< CommonBasisEvaluatorType > > >
