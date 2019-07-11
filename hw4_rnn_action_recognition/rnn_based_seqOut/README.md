0. start at ```hw4-shannon112/rnn_based_seqOut```
1. generate train_features.pt, valid_features.pt, train_vals.pt, valid_vals.pt
```
python featureExtractor.py ../../hw4_data/FullLengthVideos/videos/train  ../../hw4_data/FullLengthVideos/labels/train ./ train
python featureExtractor.py ../../hw4_data/FullLengthVideos/videos/valid  ../../hw4_data/FullLengthVideos/labels/valid ./ valid
```

2. get the best_498_loading.pth which contains optimizer and model weights

3. train model and get best.pth, log.txt, p3_curve.png
```
python train.py
then manually put them to result/
```

3. extract test feature/infos and predict labels 
```
cd ..
bash hw4_p3.sh ../hw4_data/FullLengthVideos/videos/valid ./p3_result  
```

4. evaluate ans
```
python eval_p3.py p3_result ../hw4_data/FullLengthVideos/labels/valid
```

(optional)5. plot tSNE
```
cd rnn_based_seqOut/
python visualizer.py ../p3_result/ ../../hw4_data/FullLengthVideos/labels/valid                             
```
