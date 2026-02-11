#!/bin/bash

source auto_profile/utils.sh
source auto_profile/profile_config.sh

run_docker ${TRT} ${TRT_DOCKER_IMAGE}
