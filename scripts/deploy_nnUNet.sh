#!/usr/bin/env bash
VERSION=1.02
TAG=doduo1.umcn.nl/evgenia/nnunet

docker build \
  --build-arg NNUNET_VERSION=1.6.6 \
  -t $TAG:latest \
  -t $TAG:$VERSION \
  -f /mnt/netcache/pelvis/projects/evgenia/repos/abdomenmrus-cinemri-cavity-segmentation/dockers/nnUNet/Dockerfile \
  /mnt/netcache/pelvis/projects/evgenia/repos/abdomenmrus-cinemri-cavity-segmentation

docker push $TAG:latest
docker push $TAG:$VERSION
