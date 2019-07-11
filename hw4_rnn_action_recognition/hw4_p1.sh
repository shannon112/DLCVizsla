# TODO: create shell script for Problem 1
# bash hw4_p1.sh ../hw4_data/TrimmedVideos/video/test/ ../hw4_data/TrimmedVideos/label/gt_test.csv ./
# bash hw4_p1.sh ../hw4_data/TrimmedVideos/video/valid/ ../hw4_data/TrimmedVideos/label/gt_valid.csv ./

python3 cnn_based/featureExtractor.py $1 $2 ./cnn_based test
python3 cnn_based/predict.py ./cnn_based/test_features.pt $2 cnn_based/result/best_46.pth $3
