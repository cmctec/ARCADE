#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

docker build -t grand_challenge_algorithm "$SCRIPTPATH"
