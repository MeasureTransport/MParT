from skbuild import setup

import os, sys, site

def get_install_locations():
    """Return the installation directory, or '' if no directory could be found 

       Adapted from stack overflow post https://stackoverflow.com/a/36205159
    """
    
    if '--user' in sys.argv:
        paths = (site.getusersitepackages(),)
    else:
        py_version = '%s.%s' % (sys.version_info[0], sys.version_info[1])
        paths = (s % (py_version) for s in (
            sys.prefix + '/lib/python%s/dist-packages/',
            sys.prefix + '/lib/python%s/site-packages/',
            sys.prefix + '/local/lib/python%s/dist-packages/',
            sys.prefix + '/local/lib/python%s/site-packages/',
            '/Library/Python/%s/site-packages/',
        ))

    for path in paths:
        if os.path.exists(path):
            parts = path.split('/')
            lib_indices = [index for index, item in enumerate(parts) if item == 'lib']
            return path, '/'.join(parts[0:(lib_indices[-1]+1)])
    return ''

site_folder, lib_folder = get_install_locations()


setup(
    packages=['mpart'],
    package_dir={'mpart': 'bindings/python/package'},
    package_data={'mpart':['**/*pympart*']},
    include_package_data=True,
    cmake_args=['-DKokkos_ENABLE_THREADS:BOOL=ON', f'-DSKBUILD_LIB_RPATH={lib_folder}', f'-DSKBUILD_SITE_PATH={site_folder}', '-DPYTHON_INSTALL_SUFFIX=bindings/python/package/', '-DMPART_JULIA:BOOL=OFF', '-DMPART_MATLAB:BOOL=OFF', '-DMPART_BUILD_TESTS:BOOL=OFF', '-DMPART_PYTHON:BOOL=ON', '-DPYTHON_INSTALL_PREFIX=']
)