# TODO: create shell script for Problem 3
# bash hw4_p3.sh ../hw4_data/FullLengthVideos/videos/valid ./p3_result

python3 rnn_based_seqOut/featureExtractor.py $1 ./ ./rnn_based_seqOut test
#python3 rnn_based_seqOut/predict.py rnn_based_seqOut/test_features.pt rnn_based_seqOut/test_infos.pt rnn_based_seqOut/result/best_568.pth $2
python3 rnn_based_seqOut/predict.py rnn_based_seqOut/test_features.pt rnn_based_seqOut/test_infos.pt rnn_based_seqOut/result/best_631.pth $2
