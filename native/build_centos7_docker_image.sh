#!/usr/bin/env bash

set -eu

# XXX run `docker login` first to save credentials

TAG="tom91136/centos7-llvm14:latest"

# --platform linux/amd64,linux/arm64
docker buildx build --platform linux/arm64 -t "$TAG" -t "$TAG" .
docker push "$TAG"
