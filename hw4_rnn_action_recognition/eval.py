#python eval.py p1_valid.txt ../hw4_data/TrimmedVideos/label/gt_test_ans.csv
#python eval.py p2_result.txt ../hw4_data/TrimmedVideos/label/gt_test_ans.csv
#python eval.py p2_result.txt ../hw4_data/TrimmedVideos/label/gt_valid.csv

from reader import getVideoList
import numpy as np
import sys
import os

test_predict_path = sys.argv[1]
test_label_path = sys.argv[2] #TrimmedVideos/label/gt_valid.csv

# read files
dict = getVideoList(os.path.join(test_label_path))
f = open(os.path.join(test_predict_path),'r')
predict_vals = f.read().splitlines()

# evaluation ans
print("\nevaluation ans...")
predict_vals = np.array(predict_vals).astype(int)
print("predict_vals:\n",predict_vals)
label_vals = np.array(dict['Action_labels']).astype(int)
print("label_vals:\n",label_vals)
accuracy = np.mean(predict_vals == label_vals)
print("accuracy:",accuracy)
