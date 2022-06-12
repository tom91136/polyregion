#!/usr/bin/env bash

set -eu

# XXX run `docker login` first to save credentials

TAG="tom91136/centos7-llvm14:latest"

docker build . -t "$TAG" -t "$TAG"
docker push "$TAG"
