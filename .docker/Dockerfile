FROM continuumio/miniconda3

COPY . MParT_

RUN conda env create -f MParT_/docker/environment.yml

SHELL ["conda", "run", "-n", "mpart", "/bin/bash", "-c"]

RUN ./MParT_/docker/build_kokkos.sh

RUN ./MParT_/docker/build_mpart.sh
