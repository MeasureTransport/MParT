FROM continuumio/miniconda3

COPY . MParT_

RUN --mount=type=cache,target=/opt/conda/pkgs conda env create -f MParT_/.docker/environment.yml && \
    conda clean -afy

SHELL ["conda", "run", "-n", "mpart", "/bin/bash", "-c"]

RUN git clone --depth=1 --branch 3.7.02 https://github.com/kokkos/kokkos.git && \
    mkdir kokkos/build && \
    cd kokkos/build && \
    cmake \
          -DKokkos_ENABLE_SERIAL=ON \
          -DKokkos_ENABLE_OPENMP=ON \
          -DBUILD_SHARED_LIBS=ON    \
          -DCMAKE_CXX_STANDARD=17   \
          ../ && \
    make -j$(nproc) && \
    make install && \
    cd / && \
    rm -rf kokkos

RUN cd MParT_ && \
    mkdir build && \
    cd build && \
    cmake -DMPART_BUILD_TESTS=OFF -DPYTHON_EXECUTABLE=`which python` -DMPART_FETCH_DEPS=OFF ../ && \
    make -j$(nproc) && \
    make install && \
    cd / && \
    rm -rf MParT_

ENV NB_USER=bayes \
    NB_UID=1000

RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER

WORKDIR /home/${NB_USER}
USER $NB_USER

SHELL ["/bin/bash","-c"]
RUN conda init && \
    echo 'conda activate mpart' >> ~/.bashrc

RUN echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib' >> ~/.bashrc && \
    echo 'export PYTHONPATH=$PYTHONPATH:/usr/local/python' >> ~/.bashrc && \
    echo 'export OMP_PROC_BIND=spread' >> ~/.bashrc && \
    echo 'export OMP_PLACES=threads' >> ~/.bashrc && \
    echo 'export KOKKOS_NUM_THREADS=`nproc`' >> ~/.bashrc
