==================
Template Concepts
==================

Many of the lower-level classes in MParT are templated to allow for generic implementations.  Using templates instead of other programming techniques, like virtual inheritance, makes it simpler to copy these classes to/from a GPU and can sometimes even result in more efficient CPU code.    For example, the MonotoneComponent class, which uses a generic function :math:`f(x)` to define a monotone function :math:`T_d(x)`, is templated on the type of the :math:`f` function.   It is therefore possible to construct a monotone function from any class defining :math:`f(x)` with the interface expected by the MonotoneComponent class.  The interface expected by MonotoneComponent is described by a concept.

.. topic:: Concept
    
    A concept is a set of requirements defining the interface expected by a templated function or class.



Specific concepts used throughout MParT can be found on the following pages.

.. toctree::

   concepts/cachedparameterization
   concepts/parameterizedfunction
   