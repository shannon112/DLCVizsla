# TODO: create shell script for running your GAN/ACGAN model

# bash ./hw3_p1p2.sh $1
# $1 is the folder to which you should output your fig1_2.jpg and fig2_2.jpg.

# The shell script file for running your GAN and ACGAN models.
# This script takes as input a folder and should output two images named
# fig1_2.jpg and fig2_2.jpg in the given folder.

python3 dcgan/predict.py $1 dcgan/weight_17000.pth
python3 acgan/predict.py $1 acgan/weight_99200_new_64.pth
