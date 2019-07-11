# TODO: create shell script for running your GTA model

# bash ./hw3_p4.sh $1 $2 $3
# bash ./hw3_p4.sh ../hw3_data/digits/svhn/test svhn ./test_pred.csv
# bash ./hw3_p4.sh ../hw3_data/digits/mnistm/test mnistm ./test_pred.csv
# bash ./hw3_p4.sh ../hw3_data/digits/usps/test usps ./test_pred.csv


# $1 is the directory of testing images in the target domain (e.g. hw3_data/digits/mnistm/test).
# $2 is a string that indicates the name of the target domain, which will be either mnistm, usps or svhn.
#    Note that you should run the model whose target domain corresponds with $3. For example, when $3 is mnistm, you should make your prediction using your "USPS→MNIST-M" model, NOT your "MNIST-M→SVHN" model.
# $3 is the path to your output prediction file (e.g. hw3_data/digits/mnistm/test_pred.csv).


#The shell script file for running your DANN model. This script takes as input a folder containing testing images and a string indicating the target domain, and should output the predicted results in a .csv file.


# Warning that you CANNOT USE SPACE when assign value
model_path="none"
if [ $2 == "svhn" ]
then
   model_path="gta/models_MtoS"
   python3 gta/predict.py $1 "$model_path" $3
elif [ $2 == "mnistm" ]
then
  model_path="gta/models_UtoM_org"
  python3 gta/predict_org.py $1 "$model_path" $3
elif [ $2 == "usps" ]
then
  model_path="gta/models_StoU_org"
  python3 gta/predict_org.py $1 "$model_path" $3
else
   echo "None of the condition met"
fi


# python hw3_eval.py test_pred.csv ../hw3_data/digits/svhn/test.csv
# python hw3_eval.py test_pred.csv ../hw3_data/digits/mnistm/test.csv
# python hw3_eval.py test_pred.csv ../hw3_data/digits/usps/test.csv
