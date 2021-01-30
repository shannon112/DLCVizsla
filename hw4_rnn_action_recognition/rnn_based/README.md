0. start at ```hw4-shannon112/rnn_based```
1. generate train_features.pt, valid_features.pt, train_vals.pt, valid_vals.pt
```
python featureExtractor.py ../../hw4_data/TrimmedVideos/video/train ../../hw4_data/TrimmedVideos/label/gt_train.csv ./ train
python featureExtractor.py ../../hw4_data/TrimmedVideos/video/valid ../../hw4_data/TrimmedVideos/label/gt_valid.csv ./ valid
```

2. train model and get best.pth, log.txt, p2_curve.png
```
python train.py
then manually put them to result/
```

3. extract test feature and predict
```
cd ..
bash hw4_p2.sh ../hw4_data/TrimmedVideos/video/valid/ ../hw4_data/TrimmedVideos/label/gt_valid.csv ./
(or bash hw4_p2.sh ../hw4_data/TrimmedVideos/video/test/ ../hw4_data/TrimmedVideos/label/gt_test.csv ./)
```

4. evaluate ans
```
python eval.py p2_result.txt ../hw4_data/TrimmedVideos/label/gt_valid.csv
(or python eval.py p2_result.txt ../hw4_data/TrimmedVideos/label/gt_test_ans.csv)
```

(optional)5. plot tSNE
```
cd rnn_based/
python tSNEplot.py valid_features.pt ../../hw4_data/TrimmedVideos/label/gt_valid.csv result/best.pth
```