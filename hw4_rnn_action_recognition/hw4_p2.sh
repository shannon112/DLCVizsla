# TODO: create shell script for Problem 2
# bash hw4_p2.sh ../hw4_data/TrimmedVideos/video/test/ ../hw4_data/TrimmedVideos/label/gt_test.csv ./
# bash hw4_p2.sh ../hw4_data/TrimmedVideos/video/valid/ ../hw4_data/TrimmedVideos/label/gt_valid.csv ./

python3 rnn_based/featureExtractor.py $1 $2 ./rnn_based test
python3 rnn_based/predict.py ./rnn_based/test_features.pt $2 rnn_based/result/best.pth $3
