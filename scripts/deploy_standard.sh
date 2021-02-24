#!/usr/bin/env bash
VERSION=1
TAG=doduo1.umcn.nl/evgenia/standard

docker build \
  -t $TAG:latest \
  -t $TAG:$VERSION \
  -f dockers/standard/Dockerfile \
  .

docker push $TAG:latest
docker push $TAG:$VERSION
