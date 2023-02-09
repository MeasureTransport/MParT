from skbuild import setup

setup(
    packages=['mpart'],
    package_dir={'mpart': 'bindings/python/package'},
    package_data={'mpart':['**/*pympart*']},
    include_package_data=True,
    cmake_args=['-DPYTHON_INSTALL_SUFFIX=bindings/python/package/', '-DMPART_JULIA:BOOL=OFF', '-DMPART_MATLAB:BOOL=OFF', '-DMPART_BUILD_TESTS:BOOL=OFF', '-DMPART_PYTHON:BOOL=ON', '-DPYTHON_INSTALL_PREFIX=']
)