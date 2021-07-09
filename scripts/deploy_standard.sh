#!/usr/bin/env bash
VERSION=1.04
TAG=doduo1.umcn.nl/evgenia/standard

docker build \
  -t $TAG:latest \
  -t $TAG:$VERSION \
  -f /mnt/netcache/pelvis/projects/evgenia/repos/abdomenmrus-cinemri-cavity-segmentation/dockers/standard/Dockerfile \
  /mnt/netcache/pelvis/projects/evgenia/repos/abdomenmrus-cinemri-cavity-segmentation

docker push $TAG:latest
docker push $TAG:$VERSION
