# cpp-dqn
Deep Q-Network implementation written in C++ with NNabla.

## build
Before building this repository, you need to install NNabla.
See [official instruction](https://github.com/sony/nnabla/blob/master/doc/build/build_cpp_utils.md).

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
$ ./tests/dqntest
```
