#!/bin/bash

nvidia-smi | grep 'python' | awk '{ print $3 }' | xargs -n 1 kill -9

