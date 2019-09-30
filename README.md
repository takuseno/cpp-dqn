![](https://github.com/takuseno/cpp-dqn/workflows/Build%20and%20Test/badge.svg)

# cpp-dqn
Deep Q-Network implementation written in C++ with NNabla.

This project aims for the :zap: fastest and :smile: readable DQN implementation.

macOS and Linux are currently supported.

## TODO
- [ ] reproduce [the Nature paper](https://www.nature.com/articles/nature14236).
- [ ] add more DQN-based algorithms (Double DQN, Prioritized DQN, ...)
- [ ] use [CULE](https://github.com/NVlabs/cule) for further speed up.

## third party
- [nnabla](https://github.com/sony/nnabla)
- [nnabla-ext-cuda](https://github.com/sony/nnabla-ext-cuda)
- [Arcade Learning Environment](https://github.com/mgbellemare/Arcade-Learning-Environment)
- [googletest](https://github.com/google/googletest)
- [gflags](https://github.com/gflags/gflags)
- [atari-py](https://github.com/openai/atari-py) (only for extracting ROMs)

## pull prebuilt docker container
If you want to play with this implementation on docker container, please use the prebuilt container.
```
$ docker pull takuseno/cpp-dqn
$ docker run -it --rm --runtime nvidia --name cpp-dqn takuseno/cpp-dqn:latest bash
root@834182ee578b:/cpp-dqn# ./bin/train -rom atari_roms/breakout.bin
```

DockerHub: https://hub.docker.com/r/takuseno/cpp-dqn

## build with docker
To skip manual build, use prebuilt container and mount the current directory by running the following commands.
```
$ ./scripts/up.sh --runtime nvidia
root@834182ee578b:/cpp-dqn# mkdir build
root@834182ee578b:/cpp-dqn# cd build
root@834182ee578b:/cpp-dqn/build# cmake .. -DGPU=ON
root@834182ee578b:/cpp-dqn/build# make
root@834182ee578b:/cpp-dqn/build# cd ..
root@834182ee578b:/cpp-dqn# ./bin/train -rom atari_roms/breakout.bin
```

## manual build
### nnabla
Before building this repository, you need to install NNabla.
See [official instruction](https://github.com/sony/nnabla/blob/master/doc/build/build_cpp_utils.md).
Note that arguments of cmake must be as follows.
```
$ cmake -DBUILD_CPP_UTILS=ON -DBUILD_PYTHON_PACKAGE=OFF ..
```

If you use GPU, you additionally need to install CUDA extension of NNabla.
See [official instruction](https://github.com/sony/nnabla-ext-cuda/blob/master/doc/build/build.md).

### SDL
By default, SDL libraries are used to build to render GUI. Then you need to install related libraries.
If you need to omit this, you have to set `-DUSE_SDL=OFF`.
```
# macOS
$ brew install sdl sdl_gfx sdl_image

# Ubuntu
$ sudo apt-get install libsdl1.2-dev libsdl-gfx1.2-dev libsdl-image1.2-dev
```

### build DQN
Finally, run the following codes to build DQN.
```
$ mkdir build
$ cd build
$ cmake .. # add -DGPU=ON option to build with cuda extension
$ make
```

## test
```
$ cd build
$ make
$ cd ..
$ ./scripts/test.sh
```

## format codes
`clang-format` is used to format entire codes with `llvm` style.
```
$ ./scripts/autoformat.sh
```
