# cpp-dqn
Deep Q-Network implementation written in C++ with NNabla.

## build
Before building this repository, you need to install NNabla.
See [official instruction](https://github.com/sony/nnabla/blob/master/doc/build/build_cpp_utils.md).

By default, SDL libraries are used to build to render GUI. Then you need to install related libraries.
If you need to omit this, you have to set `-DUSE_SDL=OFF`.
```
# macOS
$ brew install sdl sdl_gfx sdl_image

# Ubuntu
$ sudo apt-get install libsdl1.2-dev libsdl-gfx1.2-dev libsdl-image1.2-dev
```

Finally, run the following codes to build DQN.
```
$ mkdir build
$ cd build
$ cmake ..
$ make
```

## test
```
$ cd build
$ make
$ cd ..
$ ./build/tests/dqntest
```
