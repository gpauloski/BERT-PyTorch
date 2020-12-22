#!/bin/bash

nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -n 1 kill -9
pkill -u $USER -9 python

