#!/bin/bash -eux

if [[ "$OSTYPE" == "darwin"* ]]; then
  _DISPLAY="$(ifconfig en0 | grep "[0-9]*\.[0-9]*\.[0-9]*\.[0-9]*" | sed -e "s/ netmask .*$//g" | sed -e "s/.*inet //g"):0"
  socat TCP-LISTEN:6000,reuseaddr,fork UNIX-CLIENT:\"$DISPLAY\" &
else
  _DISPLAY=$DISPLAY
fi

sudo docker run \
  -it \
  --rm \
  -v /tmp/.X11-unix/:/tmp/.X11-unix \
  --shm-size=256m \
  -e QT_X11_NO_MITSHM=1 \
  -e DISPLAY=$_DISPLAY \
  -v ${PWD}:/cpp-dqn \
  --name cpp-dqn \
  $* \
  takuseno/cpp-dqn:latest bash
