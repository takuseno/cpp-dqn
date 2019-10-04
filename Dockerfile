FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
MAINTAINER takuseno

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

RUN apt-get update \
  && apt-get install -y --no-install-recommends ca-certificates curl wget bzip2 \
    build-essential cmake python python-pip python-setuptools libarchive-dev \
    libsdl1.2-dev libsdl-gfx1.2-dev libsdl-image1.2-dev tmux unzip \
  && pip --no-cache-dir install pyyaml mako six \
  && curl -L https://github.com/google/protobuf/archive/v3.1.0.tar.gz -o protobuf-v3.1.0.tar.gz \
  && tar xvf protobuf-v3.1.0.tar.gz \
  && cd protobuf-3.1.0 \
  && mkdir build \
  && cd build \
  && cmake -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Dprotobuf_BUILD_TESTS=OFF ../cmake \
  && make \
  && make install \
  && cd ../.. \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# build NNabla
RUN wget https://github.com/sony/nnabla/archive/v1.2.0.zip \
  && unzip v1.2.0.zip \
  && rm v1.2.0.zip \
  && mkdir nnabla-1.2.0/build \
  && cd nnabla-1.2.0/build \
  && cmake .. -DBUILD_CPP_UTILS=ON -DBUILD_PYTHON_PACKAGE=OFF -DNNABLA_UTILS_WITH_HDF5=OFF \
  && make \
  && make install \
  && cd ../..

# build CUDA extension
RUN wget https://github.com/sony/nnabla-ext-cuda/archive/v1.2.0.zip \
  && unzip v1.2.0.zip \
  && rm v1.2.0.zip \
  && mkdir nnabla-ext-cuda-1.2.0/build \
  && cd nnabla-ext-cuda-1.2.0/build \
  && cmake .. -DNNABLA_DIR=../../nnabla-1.2.0 \
     -DCPPLIB_LIBRARY=../../nnabla-1.2.0/build/lib/libnnabla.so \
     -DBUILD_PYTHON_PACKAGE=OFF \
  && make \
  && make install \
  && cd ../..

# build cpp-dqn
COPY . /cpp-dqn
RUN cd /cpp-dqn \
  && mkdir build \
  && cd build \
  && cmake .. -DGPU=ON \
  && make

WORKDIR /cpp-dqn

CMD tail -f /dev/null
