# Our image is based on Debian bullseye
FROM python:3.11-bookworm

# Set for all apt-get install, must be at the very beginning of the Dockerfile.
ENV DEBIAN_FRONTEND noninteractive

# Update the OS
# use libatlass instead of liblapack3 libblas3
RUN apt-get -y update; \
    apt-get -y install \
    build-essential \
    cython3 \
    git \
    jq \
    libatlas3-base \
    python3-astropy \
    python3-dev \
    python3-ipykernel \
    python3-ipython \
    python3-matplotlib \
    python3-numpy \
    python3-pandas \
    python3-scipy \
    python3-seaborn \
    python3-six \
    wget \
    procps \
    ; \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*; \
    apt-get -y autoremove;

# use python3 as the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# copy the repository into the container in the /mwa_qa directory
ADD . /mwa_qa
WORKDIR /mwa_qa

# install other python reqs
# RUN pip install -r requirements.txt

# install the python module in the container
RUN python -m pip install .

ENTRYPOINT bash