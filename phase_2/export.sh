#!/usr/bin/env bash

./build.sh

docker save grand_challenge_algorithm | gzip -c > grand_challenge_algorithm.tar.gz
