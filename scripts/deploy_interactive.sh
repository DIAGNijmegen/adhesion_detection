#!/usr/bin/env bash
VERSION=1
TAG=doduo1.umcn.nl/evgenia/interactive

docker build \
  -t $TAG:latest \
  -t $TAG:$VERSION \
  -f dockers/interactive/Dockerfile \
  .

docker push $TAG:latest
docker push $TAG:$VERSION
