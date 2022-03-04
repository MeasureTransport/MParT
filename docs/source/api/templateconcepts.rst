==================
Template Concepts
==================

Many of the lower-level classes in MParT are templated to allow for generic implementations.  Using templates instead of other programming techniques, like virtual inheritance, makes it simpler to copy these classes to/from a GPU and can sometimes even result in more efficient CPU code.    For example, the :code:`MonotoneComponent`` class, which uses a generic function :math:`f(x)` to define a monotone function :math:`T_d(x)`, is templated on the type of the :math:`f` function.   It is therefore possible to construct a monotone function from any class defining :math:`f(x)`, as long as the class contains the functions (i.e., the interface) expected by :code:`MonotoneComponent`.  In the language of generic programming, the necessary interface is a specific `concept <https://en.wikipedia.org/wiki/Concept_(generic_programming)>`_.

.. topic:: Concept
    
    A concept is a set of requirements defining the interface expected by a templated function or class.


Specific concepts used throughout MParT can be found on the following pages.

.. toctree::

   concepts/cachedparameterization
   concepts/parameterizedfunction
   