# TODO: create shell script for running your YoloV1-vgg16bn model
# usage: bash ./hw2.sh $1 $2

#python predict.py ../hw2_train_val/val1500/images/ ../hw2_train_val/val1500/labelpre/
wget -O model.pth https://www.dropbox.com/s/l7ufbg3wdhm2n5w/best_baseline.pth?raw=1
python3 predict.py $1 $2 model.pth
