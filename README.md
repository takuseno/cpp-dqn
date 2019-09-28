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
- [nnabla](https://github.com/sony/nnabla) (build this by yourself)
- [nnabla-ext-cuda](https://github.com/sony/nnabla-ext-cuda) (build this by yourself)
- [Arcade Learning Environment](https://github.com/mgbellemare/Arcade-Learning-Environment)
- [googletest](https://github.com/google/googletest)
- [gflags](https://github.com/gflags/gflags)
- [atari-py](https://github.com/openai/atari-py) (only for extracting ROMs)

## build
### nnabla
Before building this repository, you need to install NNabla.
See [official instruction](https://github.com/sony/nnabla/blob/master/doc/build/build_cpp_utils.md).

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
