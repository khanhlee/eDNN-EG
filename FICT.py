"""
	@author   : Khanh Lee
	@Major    : Computer Science
	@Language : Python 3.5.3
	@Desc     : FastText to classify protein sequences
	References: https://fasttext.cc/docs/en/support.html
				https://fasttext.cc/docs/en/supervised-tutorial.html
				https://github.com/facebookresearch/fastText/
				https://pypi.org/project/fasttext/#description
				https://www.tutorialkart.com/fasttext/make-model-learn-word-representations-using-fasttext-python/
				https://blog.manash.me/how-to-use-pre-trained-word-vectors-from-facebooks-fasttext-a71e6d55f27
				https://github.com/facebookresearch/fastText
"""

"""
# Required Python Package
import os
import pandas as pd
import time


class BasicFunctions:
    '''
        Constructor
    '''
    def __init__(self):
        print("Start your function...");
    
    
    '''
        This function is used to read all your fasta files in dir and
        store all fasta id and seq in dataframe.
    '''
    def readFastaInDir(self, inputDir):
        fastaDir = inputDir;
        # Error handling
        try:
            # Set your time
            start_time = time.clock()
            # Create new dataframe to store
            COLUMN_NAMES=['ID', 'sequence']
            COLLECT = pd.DataFrame(columns=COLUMN_NAMES)
            
            # Loop all files
            for root, dirs, files in os.walk(fastaDir):
                i=0;
                for file in files:
                    filePath=os.path.join(root,file)
                    #print("File {0}: {1}".format(i+1, filePath));
                    # Read file
                    f = open(filePath, "r")
                    fastaName = "";
                    fastaSeq = "";
                    count=1;
                    for line in f:
                        if count==1:
                            try:
                                fastaName = line[2:]
                            except ValueError:
                                print("Error fasta name: ",filePath);
                        
                        elif count==2:
                            fastaSeq = line.replace('\n', '');
                            
                        count=count+1;
                        
                    #print(fastaName);
                    #print(fastaSeq);
                    
                    # Insert to data frame
                    COLLECT.loc[i] = [fastaName, fastaSeq]
                    
                    i=i+1;    
                    f.close
                
                # Print total file
                print("--Time: {0} seconds, Total Fasta files: {1}--".format(round((time.clock() - start_time),4), i));
        
        except:
            print("Error message: You got an error.")

        return COLLECT;
    


    
    
'''
Test Functions with Independent Program
Allows your program to be run by programs that import it
'''
if __name__ == '__main__':
    # Call class
    x = BasicFunctions()
    
    print("====================================================================");
    # Set dir, call function and read all fasta fules
    print('Read fasta files, please wait ...')
    
    # Find max length: get the length of the string of column in a dataframe
    #dataFasta['Length'] = dataFasta['sequence'].apply(len)
    #print(dataFasta)
    
    # Get max length of protein sequences
    #max(dataFasta['Length'].tolist())
    #min(dataFasta['Length'].tolist())

    
    #==========================================================================
    # MAKE TRAINING DATA
    #==========================================================================    
    # GET DATA CLASS A (ion channels)
    fastaDir = "D:/LAB PROJECT/[Journal 2 - 2018] Ion transporter and Ion Channel from membrane proteins/DATA/3 All FASTA and PSSM/FASTA/ionchannels/train/";
    dataFasta = x.readFastaInDir(fastaDir);
    
    indexCol = 0
    className = "__label__A"  # can be a list, a Series, an array or a scalar   
    dataFasta.insert(loc=indexCol, column='class', value=className)
    
    GET_DATA_A = dataFasta[['class','sequence']]
    
    
    # GET DATA CLASS B (ion transporters)
    fastaDir = "D:/LAB PROJECT/[Journal 2 - 2018] Ion transporter and Ion Channel from membrane proteins/DATA/3 All FASTA and PSSM/FASTA/iontransporters/train/";
    dataFasta = x.readFastaInDir(fastaDir);
    
    indexCol = 0
    className = "__label__B"  # can be a list, a Series, an array or a scalar   
    dataFasta.insert(loc=indexCol, column='class', value=className)
    
    GET_DATA_B = dataFasta[['class','sequence']]
    
    
    # GET DATA CLASS C (membrane proteins)
    fastaDir = "D:/LAB PROJECT/[Journal 2 - 2018] Ion transporter and Ion Channel from membrane proteins/DATA/3 All FASTA and PSSM/FASTA/membraneproteins/train/";
    dataFasta = x.readFastaInDir(fastaDir);
    
    indexCol = 0
    className = "__label__C"  # can be a list, a Series, an array or a scalar   
    dataFasta.insert(loc=indexCol, column='class', value=className)
    
    GET_DATA_C = dataFasta[['class','sequence']]
    
    
    
    # Make train_A (ion channel - membrane proteins)
    train_A = pd.concat([GET_DATA_A,GET_DATA_C]);
    
    # Make train_B (ion transporters - membrane proteins)
    train_B = pd.concat([GET_DATA_B,GET_DATA_C]);
    
    # Make train_C (ion transporters - ion channel)
    train_C = pd.concat([GET_DATA_B,GET_DATA_A]);
    
    
    
    WorkDir = "D:/LAB PROJECT/[Journal 2 - 2018] Ion transporter and Ion Channel from membrane proteins/DATA/5 fasttext data/DATA/";
    output_file = WorkDir+"A_trainChannelsMembrane.csv";
    train_A.to_csv(output_file, 
                    encoding='utf-8', 
                    sep=' ',             # character, default ‘,’
                    index=False,
                    header=False
                    ) 
    
    output_file = WorkDir+"B_trainTransportersMembrane.csv";
    train_B.to_csv(output_file, 
                    encoding='utf-8', 
                    sep=' ',             # character, default ‘,’
                    index=False,
                    header=False
                    ) 

    output_file = WorkDir+"C_trainTransportersChannels.csv";
    train_C.to_csv(output_file, 
                    encoding='utf-8', 
                    sep=' ',             # character, default ‘,’
                    index=False,
                    header=False
                    ) 


    
    #==========================================================================
    # MAKE TESTING DATA
    #==========================================================================    
    # GET DATA CLASS A (ion channels)
    fastaDir = "D:/LAB PROJECT/[Journal 2 - 2018] Ion transporter and Ion Channel from membrane proteins/DATA/3 All FASTA and PSSM/FASTA/ionchannels/test/";
    dataFasta = x.readFastaInDir(fastaDir);
    
    indexCol = 0
    className = "__label__A"  # can be a list, a Series, an array or a scalar   
    dataFasta.insert(loc=indexCol, column='class', value=className)
    
    GET_DATA_A = dataFasta[['class','sequence']]
    
    
    # GET DATA CLASS B (ion transporters)
    fastaDir = "D:/LAB PROJECT/[Journal 2 - 2018] Ion transporter and Ion Channel from membrane proteins/DATA/3 All FASTA and PSSM/FASTA/iontransporters/test/";
    dataFasta = x.readFastaInDir(fastaDir);
    
    indexCol = 0
    className = "__label__B"  # can be a list, a Series, an array or a scalar   
    dataFasta.insert(loc=indexCol, column='class', value=className)
    
    GET_DATA_B = dataFasta[['class','sequence']]
    
    
    # GET DATA CLASS C (membrane proteins)
    fastaDir = "D:/LAB PROJECT/[Journal 2 - 2018] Ion transporter and Ion Channel from membrane proteins/DATA/3 All FASTA and PSSM/FASTA/membraneproteins/test/";
    dataFasta = x.readFastaInDir(fastaDir);
    
    indexCol = 0
    className = "__label__C"  # can be a list, a Series, an array or a scalar   
    dataFasta.insert(loc=indexCol, column='class', value=className)
    
    GET_DATA_C = dataFasta[['class','sequence']]
    
    
    
    # Make test_A (ion channel - membrane proteins)
    test_A = pd.concat([GET_DATA_A,GET_DATA_C]);
    
    # Make test_B (ion transporters - membrane proteins)
    test_B = pd.concat([GET_DATA_B,GET_DATA_C]);
    
    # Make test_C (ion transporters - ion channel)
    test_C = pd.concat([GET_DATA_B,GET_DATA_A]);
    
    
    
    WorkDir = "D:/LAB PROJECT/[Journal 2 - 2018] Ion transporter and Ion Channel from membrane proteins/DATA/5 fasttext data/DATA/";
    output_file = WorkDir+"A_testChannelsMembrane.csv";
    test_A.to_csv(output_file, 
                    encoding='utf-8', 
                    sep=' ',             # character, default ‘,’
                    index=False,
                    header=False
                    ) 
    
    output_file = WorkDir+"B_testTransportersMembrane.csv";
    test_B.to_csv(output_file, 
                    encoding='utf-8', 
                    sep=' ',             # character, default ‘,’
                    index=False,
                    header=False
                    ) 

    output_file = WorkDir+"C_testTransportersChannels.csv";
    test_C.to_csv(output_file, 
                    encoding='utf-8', 
                    sep=' ',             # character, default ‘,’
                    index=False,
                    header=False
                    ) 

"""











# =====================================================================================
# NOTE: Separate above code
# =====================================================================================



def manual_confusion_matrix(test_label_original, test_label_predicted, label_pos, label_neg, saveFileName):
	# Term for confusion matrix
	TP=0.0;FP=0.0;FN=0.0;TN=0.0;
	
	for count in range(0, len(test_label_original)):
		predictedClass = test_label_predicted[count]
		expectedClass = test_label_original[count];

		# Basic calculation of confusion matrix 
		if predictedClass == label_pos and expectedClass == label_pos:
			TP=TP+1;

		if predictedClass == label_pos and expectedClass == label_neg:
			FP=FP+1;

		if predictedClass == label_neg and expectedClass == label_pos:
			FN=FN+1;

		if predictedClass == label_neg and expectedClass == label_neg:
			TN=TN+1;
	
	Sensitivity=0.0; Specificity=0.0;Precision=0.0;Accuracy=0.0;
	MCC=0.0;F1=0.0;
	try:
		Sensitivity=(TP/(TP+FN));
		Specificity=TN/(FP+TN);
		Precision=TP/(TP+FP);
		Accuracy = (TP+TN)/(TP+FP+TN+FN);
		MCC=((TP*TN)-(FP*FN))/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN));
		F1=(2*TP)/((2*TP)+FP+FN);
	
	except:
		Sensitivity=0.0; Specificity=0.0;Precision=0.0;Accuracy=0.0;
		MCC=0.0;F1=0.0;
	
	# Print confusion matrix terms
	print("=============================================");
	print("Confusion matrix for a binary my_predictor:");
	print("(number of) true positive (TP): ", TP)
	print("(number of) false positive (FP): ", FP)
	print("(number of) false negative (FN): ", FN)
	print("(number of) true negative (TN): ", TN)

	# Performance Evaluations
	print("Sensitivity: {0}%".format(round(Sensitivity, 4)*100))
	print("Specificity: {0}%".format(round(Specificity, 4)*100))
	print("Precision: {0}%".format(round(Precision, 4)*100))
	print("Accuracy: {0}%".format(round(Accuracy, 4)*100))
	print("MCC: {0}%".format(round(MCC, 4)*100))
	print("F1: {0}%".format(round(F1, 4)*100))
	print("=============================================\n");
		
	
	## Store result in a file
	#results = "epoch={8},TP={0}, FP={1}, FN={2}, TN={3}, Sensitivity={4}, Specificity={5}, Accuracy={6}, MCC={7}".format(TP,FP,FN,TN,Sensitivity,Specificity,Accuracy,MCC,number_epoch);
	#fileName = saveFileName;
	#readfile = open(fileName, "a")
	#if(os.path.getsize(fileName) > 0):
	#   readfile.write("\n"+results)
	#else:
	#   readfile.write(results)
    #
	#readfile.close()
	

	
	
print('========================================')
print("IMPLEMENTATION OF FACEBOOK'S FASTTEXT")
print('========================================')

import pandas as pd
import numpy as np
import fasttext
from sklearn.metrics import confusion_matrix
import math

## Skipgram model
#model = fasttext.skipgram('testsemmy.csv', 'model')
#print (model.words) # list of words in dictionary
#
## CBOW model
#model = fasttext.cbow('testsemmy.csv', 'model')
#print(model.words) # list of words in dictionary

#model = fasttext.load_model('model.bin')
#print (model.words) # list of words in dictionary
#print (model['AGD']) # get the vector of the word 'king'


# =======================================================================
# FILE NAME AND SOME PARAMENTERS
# =======================================================================
PATH_FILE_TRAIN = 'data/training.csv';
PATH_FILE_TEST = "data/testing.csv";
PATH_FILE_OUTPUT_MODEL = 'model';
y_pos = 'A';
y_neg = 'B';

# =======================================================================
# LOAD TRAINING DATA AND TRAIN THE MODEL
# =======================================================================
# Text classification for predicting the protein functions
my_predictor = fasttext.supervised(
					PATH_FILE_TRAIN, 					# Training data file
					output=PATH_FILE_OUTPUT_MODEL, 		# Output file path
					label_prefix='__label__', 			# Required for label
					#ws=3,								# size of the context window [5]
					epoch=100,							# number of epochs [default: 5]
					lr=1.0,								# learning rate [default: 0.05]
					dim=300,							# size of word vectors [default: 100]
					loss='ns', 							# loss function {ns, hs, softmax} [softmax]
					word_ngrams=4,						# max length of word ngram [default: 1 or unigram]
					bucket= 2000000)					# number of buckets [2000000]

#result = my_predictor.test(PATH_FILE_TEST)
#print ('P@1:', result.precision)
#print ('R@1:', result.recall)
#print ('Number of examples:', result.nexamples)


# =======================================================================
# LOAD TESTING DATA
# =======================================================================
df1 = pd.read_csv(PATH_FILE_TEST, sep=" ", usecols=[0, 1], header=None)

# Get Original Class
ori_class = [className.replace('__label__','') for className in df1.iloc[:,0].values]
#print(ori_class);

# Conver list to numpy array
myarray = np.asarray(df1.iloc[:,1].values)
#print(myarray.flatten())

#Python convert tuple to array 
myarray_default = myarray.flatten();

# Or with probability
#labels = my_predictor.predict_proba(myarray_default)
#print("Prediction probability: ",labels[0]);

#print ([ labels[index][0] for index in range(len(labels))])

# Predict label only
labels = my_predictor.predict(myarray_default)
#print("Prediction2: ",labels);

# Get Predicted Class
predicted_class = [ labels[index][0] for index in range(len(labels))];
#print ([ labels[index][0] for index in range(len(labels))])




# =======================================================================
# RESULT: CONFUSION MATRIX
# =======================================================================
# Save: cofused Matrix
TN, FP, FN, TP = confusion_matrix(ori_class, predicted_class, labels=[y_neg, y_pos]).ravel();

Sensitivity=0.0; Specificity=0.0;Precision=0.0;Accuracy=0.0;
MCC=0.0;F1=0.0;
try:
	Sensitivity=round((TP/(TP+FN)), 4);
	Specificity=round(TN/(FP+TN), 4);
	Precision=round(TP/(TP+FP), 4);
	Accuracy = round((TP+TN)/(TP+FP+TN+FN), 4);
	MCC=round(((TP*TN)-(FP*FN))/(math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))), 4);
	F1=round((2*TP)/((2*TP)+FP+FN), 4);

except:
	Sensitivity=0.0; Specificity=0.0;Precision=0.0;Accuracy=0.0;
	MCC=0.0;F1=0.0;

print('TP: {0}, FP: {1}, TN: {2}, FN: {3}, Sen: {4}, Spe: {5}, Acc: {6} \n'.format(TP,FP,TN,FN,Sensitivity, Specificity, Accuracy))
			

			
# Manual Confusion matrix and save
output_file = "";
manual_confusion_matrix(ori_class, predicted_class, y_pos, y_neg, output_file)




# =======================================================================
# LOAD PRE-TRAINED MODEL
# =======================================================================
#File .bin that previously trained or generated by fastText can be loaded using this function

#model = fasttext.load_model('deep_ion_model.bin', encoding='utf-8')



