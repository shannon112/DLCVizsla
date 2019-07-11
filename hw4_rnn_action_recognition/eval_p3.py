#python eval_p3.py p3_result ../hw4_data/FullLengthVideos/labels/valid

import numpy as np
import sys
import os
import glob

test_predict_path = sys.argv[1]
test_label_path = sys.argv[2] #TrimmedVideos/label/gt_valid.csv

predict_filenames = sorted(glob.glob(os.path.join(test_predict_path, '*.txt')))
label_filenames = sorted(glob.glob(os.path.join(test_label_path, '*.txt')))
videodir_names = [label_filename.split('/')[-1] for label_filename in label_filenames]

mean_accuracy = 0
for i,(predict_filename,label_filename) in enumerate(zip(predict_filenames,label_filenames)):
    # read predict files
    f = open(os.path.join(predict_filename),'r')
    predict_vals = f.read().splitlines()

    # read label files
    f = open(os.path.join(label_filename),'r')
    label_vals = f.read().splitlines()

    # evaluation ans
    print("\n"+videodir_names[i]+" evaluation ans...")
    predict_vals = np.array(predict_vals).astype(int)
    #print("predict_vals:\n",predict_vals)
    label_vals = np.array(label_vals).astype(int)
    #print("label_vals:\n",label_vals)
    accuracy = np.mean(predict_vals == label_vals)
    print("accuracy:",accuracy)
    mean_accuracy += accuracy
mean_accuracy /= len(label_filenames)
print("\nmean_accuracy:",mean_accuracy)
