# Docker file for predicting if a reservation is likely to be cancelled by given a hotel booking detail
# author: Chen Zhao
# contributors: Jared Splinter, Debananda Sarkar, Peter Yang
# date: 2020-12-11

# use rocker/tidyverse as the base image
FROM rocker/tidyverse

RUN apt-get update

# install the kableExtra package using install.packages
RUN Rscript -e "install.packages('kableExtra')"

# install the anaconda distribution of python
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -a -y && \
    /opt/conda/bin/conda update -n base -c defaults conda

# put anaconda python in path
ENV PATH="/opt/conda/bin:${PATH}"

# install docopt python package
RUN conda install -y -c anaconda \ 
    docopt \
    requests

# install altair
RUN conda install -y -c conda-forge altair vega_datasets

# install altair_saver
RUN conda install -y -c conda-forge altair_saver
