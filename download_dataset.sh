#!/bin/bash

wget --no-check-certificate https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip \
-O ./cats_and_dogs_filtered.zip

unzip ./cats_and_dogs_filtered.zip