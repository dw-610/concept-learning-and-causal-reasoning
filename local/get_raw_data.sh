#!/bin/bash

# download the datasets
printf "\nDownloading the training images..."
curl -o train.zip https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip
printf "\nDownloading the test images..."
curl -o test.zip https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip
printf "\nDownloading the test labels..."
curl -o test_labels.zip https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip

# make the necessary directories
mkdir raw_data
mkdir raw_data/train
mkdir raw_data/train/images
mkdir raw_data/train/anno
mkdir raw_data/test
mkdir raw_data/test/images

# unzip the files
printf "\nUnzipping the training images..."
unzip -qq train.zip
printf "\nUnzipping the test images..."
unzip -qq test.zip
printf "\nUnzipping the test labels..."
unzip -qq test_labels.zip

# move the files to the correct directories
mv GTSRB/Final_Training/Images/* raw_data/train/images/
mv GTSRB/Final_Test/Images/*.ppm raw_data/test/images
mv GT-final_test.csv raw_data/test/anno.csv
mv raw_data/train/images/*/*.csv raw_data/train/anno/

# cleanup
rm -r GTSRB
rm *.zip

printf "\nDone."