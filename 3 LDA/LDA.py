import tensorflow as tf
import numpy as np
import time

from data_util import load_data_multilabel_new,load_data_multilabel_new_k_fold,create_voabulary,create_voabulary_label
from tflearn.data_utils import to_categorical, pad_sequences
import os
import sys
import word2vec
import pickle
import random as rn
import statistics
import warnings

import gensim
import gensim.corpora as corpora
from pprint import pprint
from sklearn.metrics.pairwise import cosine_similarity

#start time
start_time = time.time()

#only using CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#notification/warning settings
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #block tensorflow logs
warnings.filterwarnings("ignore", category=UserWarning) #block sklearn UserWarning

FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("dataset","bibsonomy-clean","dataset to chose") # two options: "bibsonomy-clean" and "zhihu-sample"

tf.app.flags.DEFINE_integer("sequence_length",300,"max sentence length")
tf.app.flags.DEFINE_integer("embed_size",100,"embedding size")

#for simulating missing labels
tf.app.flags.DEFINE_float("keep_label_percent",1,"the percentage of labels in each instance of the training data to be randomly reserved, the rest labels are dropped to simulate the missing label scenario.")

#for both tuning and final testing
tf.app.flags.DEFINE_string("training_data_path_bib","../datasets/bibsonomy_preprocessed_merged_final.txt","path of traning data.") # for bibsonomy dataset
tf.app.flags.DEFINE_string("training_data_path_zhihu","../datasets/question_train_set_cleaned_150000.txt","path of traning data.") # for zhihu dataset
tf.app.flags.DEFINE_string("training_data_path_cua","../datasets/citeulike_a_cleaned_th10.txt","path of traning data.") # for cua dataset
tf.app.flags.DEFINE_string("training_data_path_cut","../datasets/citeulike_t_cleaned_th10.txt","path of traning data.") # for cut dataset

tf.app.flags.DEFINE_string("word2vec_model_path_bib","../embeddings/word-bib.bin-100","word2vec's vocabulary and vectors for inputs")
tf.app.flags.DEFINE_string("word2vec_model_path_zhihu","../embeddings/word150000.bin-100","word2vec's vocabulary and vectors")
tf.app.flags.DEFINE_string("word2vec_model_path_cua","../embeddings/word-citeulike-a.bin-100","word2vec's vocabulary and vectors")
tf.app.flags.DEFINE_string("word2vec_model_path_cut","../embeddings/word-citeulike-t-th10.bin-100","word2vec's vocabulary and vectors")

tf.app.flags.DEFINE_float("valid_portion",0.1,"dev set or test set portion") # this is only valid when kfold is -1, which means we hold out a fixed set for validation. 
# if we set this as 0.1, then there will be 0.81 0.09 0.1 for train-valid-test split (same as the split of 10-fold cross-validation); if we set this as 0.111, then there will be 0.8 0.1 0.1 for train-valid-test split.
tf.app.flags.DEFINE_float("test_portion",0.1,"held-out evaluation: test set portion")
tf.app.flags.DEFINE_integer("kfold",10,"k-fold cross-validation") # if k is -1, then not using kfold cross-validation

tf.app.flags.DEFINE_string("marking_id","","an marking_id (or group_id) for better marking: will show in the output filenames")

tf.app.flags.DEFINE_boolean("multi_label_flag",True,"use multi label or single label.")
tf.app.flags.DEFINE_boolean("report_rand_pred",True,"report prediction for qualitative analysis")
tf.app.flags.DEFINE_float("ave_labels_per_doc",11.59,"average labels per document for bibsonomy dataset")
#tf.app.flags.DEFINE_integer("topk",5,"using top-k predicted labels for evaluation")

tf.app.flags.DEFINE_string("mallet_path",r'C:\mallet-2.0.8\bin\mallet',"MALLET path for the gensim MALLET wrapper")
tf.app.flags.DEFINE_integer("num_topics",100,"number of topics for LDA")
#tf.app.flags.DEFINE_float("alpha",50/num_topics,"the hyperparameter alpha for LDA")
tf.app.flags.DEFINE_integer("iterations",1000,"number of iterations for LDA")
tf.app.flags.DEFINE_integer("k_num_doc",3,"number of most similar documents")

# 0. configuration
# todo: add command-line argument

def main(_):
    #os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    if FLAGS.dataset == "bibsonomy-clean":
        word2vec_model_path = FLAGS.word2vec_model_path_bib
        traning_data_path = FLAGS.training_data_path_bib
        FLAGS.sequence_length = 300
        FLAGS.ave_labels_per_doc = 11.59
        
    elif FLAGS.dataset == "zhihu-sample":
        word2vec_model_path = FLAGS.word2vec_model_path_zhihu
        traning_data_path = FLAGS.training_data_path_zhihu
        FLAGS.sequence_length = 100
        FLAGS.ave_labels_per_doc = 2.45
        
    elif FLAGS.dataset == "citeulike-a-clean":
        word2vec_model_path = FLAGS.word2vec_model_path_cua
        traning_data_path = FLAGS.training_data_path_cua
        FLAGS.sequence_length = 300
        FLAGS.ave_labels_per_doc = 11.6
        
    elif FLAGS.dataset == "citeulike-t-clean":
        word2vec_model_path = FLAGS.word2vec_model_path_cut
        traning_data_path = FLAGS.training_data_path_cut
        FLAGS.sequence_length = 300
        FLAGS.ave_labels_per_doc = 7.68
        
    # 1. create trainlist, validlist and testlist 
    trainX, trainY, testX, testY = None, None, None, None
    vocabulary_word2index, vocabulary_index2word = create_voabulary(word2vec_model_path,name_scope=FLAGS.dataset + "-lda") #simple='simple'
    vocabulary_word2index_label,vocabulary_index2word_label = create_voabulary_label(voabulary_label=traning_data_path, name_scope=FLAGS.dataset + "-lda")
    num_classes=len(vocabulary_word2index_label)
    print(vocabulary_index2word_label[0],vocabulary_index2word_label[1])

    vocab_size = len(vocabulary_word2index)
    print("vocab_size:",vocab_size)

    # choosing whether to use k-fold cross-validation or hold-out validation
    if FLAGS.kfold == -1: # hold-out
        train, valid, test = load_data_multilabel_new(vocabulary_word2index, vocabulary_word2index_label,keep_label_percent=FLAGS.keep_label_percent,valid_portion=FLAGS.valid_portion,test_portion=FLAGS.test_portion,multi_label_flag=FLAGS.multi_label_flag,traning_data_path=traning_data_path) 
        # here train, test are tuples; turn train into trainlist.
        trainlist, validlist, testlist = list(), list(), list()
        trainlist.append(train)
        validlist.append(valid)
        testlist.append(test)
    else: # k-fold
        trainlist, validlist, testlist = load_data_multilabel_new_k_fold(vocabulary_word2index, vocabulary_word2index_label,keep_label_percent=FLAGS.keep_label_percent,kfold=FLAGS.kfold,test_portion=FLAGS.test_portion,multi_label_flag=FLAGS.multi_label_flag,traning_data_path=traning_data_path)
        # here trainlist, testlist are list of tuples.
    # get and pad testing data: there is only one testing data, but kfold training and validation data
    assert len(testlist) == 1
    testX, testY = testlist[0]
    testX = pad_sequences(testX, maxlen=FLAGS.sequence_length, value=0.)  # padding to max length

    # 3. transform trainlist to the format. x_train, x_test: training and test feature matrices of size (n_samples, n_features)
    #print(len(trainlist))
    #trainX,trainY = trainlist[0]
    #trainX = pad_sequences(trainX, maxlen=FLAGS.sequence_length, value=0.)
    #print(len(trainX))
    #print(len(trainX[0]))
    #print(trainX[0])
    #print(len(trainY))
    #print(len(trainY[0]))
    #print(trainY[0])
    #print(np.asarray(trainY).shape)
    
    num_runs = len(trainlist)
    #validation results variables
    valid_acc_th,valid_prec_th,valid_rec_th,valid_fmeasure_th,valid_hamming_loss_th =[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs # initialise the result lists
    final_valid_acc_th,final_valid_prec_th,final_valid_rec_th,final_valid_fmeasure_th,final_valid_hamming_loss_th = 0.0,0.0,0.0,0.0,0.0
    min_valid_acc_th,min_valid_prec_th,min_valid_rec_th,min_valid_fmeasure_th,min_valid_hamming_loss_th = 0.0,0.0,0.0,0.0,0.0
    max_valid_acc_th,max_valid_prec_th,max_valid_rec_th,max_valid_fmeasure_th,max_valid_hamming_loss_th = 0.0,0.0,0.0,0.0,0.0
    std_valid_acc_th,std_valid_prec_th,std_valid_rec_th,std_valid_fmeasure_th,std_valid_hamming_loss_th = 0.0,0.0,0.0,0.0,0.0
    #testing results variables
    test_acc_th,test_prec_th,test_rec_th,test_fmeasure_th,test_hamming_loss_th = [0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs # initialise the testing result lists
    final_test_acc_th,final_test_prec_th,final_test_rec_th,final_test_fmeasure_th,final_test_hamming_loss_th = 0.0,0.0,0.0,0.0,0.0
    min_test_acc_th,min_test_prec_th,min_test_rec_th,min_test_fmeasure_th,min_test_hamming_loss_th = 0.0,0.0,0.0,0.0,0.0
    max_test_acc_th,max_test_prec_th,max_test_rec_th,max_test_fmeasure_th,max_test_hamming_loss_th = 0.0,0.0,0.0,0.0,0.0
    std_test_acc_th,std_test_prec_th,std_test_rec_th,std_test_fmeasure_th,std_test_hamming_loss_th = 0.0,0.0,0.0,0.0,0.0
    #output variables
    output_valid = ""
    output_test = ""
    output_csv_valid = "fold,hamming_loss,acc,prec,rec,f1"
    output_csv_test = "fold,hamming_loss,acc,prec,rec,f1"
        
    time_train = [0]*num_runs # get time spent in training    
    num_run = 0
    
    mallet_path = FLAGS.mallet_path
    num_topics = FLAGS.num_topics
    alpha = 50/num_topics
    iterations = FLAGS.iterations
    k_num_doc = FLAGS.k_num_doc
    
    remove_pad_id = True
    remove_dot = True
    docs_test = generateLDAdocFromIndex(testX,vocabulary_index2word,remove_pad_id=remove_pad_id,remove_dot=remove_dot)
    
    for trainfold in trainlist:
        # get training and validation data
        trainX,trainY=trainfold
        trainX = pad_sequences(trainX, maxlen=FLAGS.sequence_length, value=0.)
        # generate training data for gensim MALLET wrapper for LDA
        docs = generateLDAdocFromIndex(trainX,vocabulary_index2word,remove_pad_id=remove_pad_id,remove_dot=remove_dot)
        #print(docs[10])
        id2word = corpora.Dictionary(docs)
        corpus = [id2word.doc2bow(text) for text in docs]
        #print(corpus[10])
        # generate validation data for gensim MALLET wrapper for LDA
        validX,validY=validlist[num_run]
        validX = pad_sequences(validX, maxlen=FLAGS.sequence_length, value=0.)
        docs_valid = generateLDAdocFromIndex(validX,vocabulary_index2word,remove_pad_id=remove_pad_id,remove_dot=remove_dot)
        corpus_valid = [id2word.doc2bow(text) for text in docs_valid]
        # generate testing data for gensim MALLET wrapper for LDA
        corpus_test = [id2word.doc2bow(text) for text in docs_test]
        
        # training
        start_time_train = time.time()
        print('start training fold',str(num_run))
        
        model = gensim.models.wrappers.LdaMallet(mallet_path,corpus=corpus,num_topics=num_topics,alpha=alpha,id2word=id2word,iterations=iterations)
        pprint(model.show_topics(formatted=False))
        
        print('num_run',str(num_run),'train done.')
        
        time_train[num_run] = time.time() - start_time_train
        print("--- training of fold %s took %s seconds ---" % (num_run,time_train[num_run]))
        
        # represent each document as a topic vector
        #mat_train = np.array(model[corpus]) # this will cause an Error with large num_topics, e.g. 1000 or higher.
        #Thus, we turn the MALLET LDA model to a native Gensim LDA model
        model = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(model)
        mat_train = np.array(model.get_document_topics(corpus,minimum_probability=0.0))
        #print(len(model[corpus[0]]))
        #print(len(model[corpus[1]]))
        #print(len(model[corpus[2]]))
        #print(mat_train.shape)
        mat_train = mat_train[:,:,1] # documents in training set as a matrix of topic probabilities
    
        # evaluate on training data
        #if num_run == 0 and FLAGS.kfold != -1: # do this only for the first fold in k-fold cross-validation to save time
        #    acc, prec, rec, f_measure, hamming_loss = do_eval_lda(model, k_num_doc, mat_train, trainY, corpus, trainY, vocabulary_index2word_label, hamming_q=FLAGS.ave_labels_per_doc)
        #    print('training:', acc, prec, rec, f_measure, hamming_loss)
        
        # validation
        valid_acc_th[num_run],valid_prec_th[num_run],valid_rec_th[num_run],valid_fmeasure_th[num_run],valid_hamming_loss_th[num_run] = do_eval_lda(model, k_num_doc, mat_train, trainY, corpus_valid, validY, vocabulary_index2word_label, hamming_q=FLAGS.ave_labels_per_doc)
        print("LDA==>Run %d Validation Accuracy: %.3f\tValidation Hamming Loss: %.3f\tValidation Precision: %.3f\tValidation Recall: %.3f\tValidation F-measure: %.3f" % (num_run,valid_acc_th[num_run],valid_hamming_loss_th[num_run],valid_prec_th[num_run],valid_rec_th[num_run],valid_fmeasure_th[num_run]))
        output_valid = output_valid + "\n" + "LDA==>Run %d Validation Accuracy: %.3f\tValidation Hamming Loss: %.3f\tValidation Precision: %.3f\tValidation Recall: %.3f\tValidation F-measure: %.3f" % (num_run,valid_acc_th[num_run],valid_hamming_loss_th[num_run],valid_prec_th[num_run],valid_rec_th[num_run],valid_fmeasure_th[num_run]) + "\n" # also output the results of each run.
        output_csv_valid = output_csv_valid + "\n" + str(num_run) + "," + str(valid_hamming_loss_th[num_run]) + "," + str(valid_acc_th[num_run]) + "," + str(valid_prec_th[num_run]) + "," + str(valid_rec_th[num_run]) + "," + str(valid_fmeasure_th[num_run])
        
        start_time_test = time.time()
        # evaluate on testing data
        test_acc_th[num_run],test_prec_th[num_run],test_rec_th[num_run],test_fmeasure_th[num_run],test_hamming_loss_th[num_run] = do_eval_lda(model, k_num_doc, mat_train, trainY, corpus_test, testY, vocabulary_index2word_label, hamming_q=FLAGS.ave_labels_per_doc)
        print("LDA==>Run %d Test Accuracy: %.3f\tTest Hamming Loss: %.3f\tTest Precision: %.3f\tTest Recall: %.3f\tTest F-measure: %.3f" % (num_run,test_acc_th[num_run],test_hamming_loss_th[num_run],test_prec_th[num_run],test_rec_th[num_run],test_fmeasure_th[num_run]))
        output_test = output_test + "\n" + "LDA==>Run %d Test Accuracy: %.3f\tTest Hamming Loss: %.3f\tTest Precision: %.3f\tTest Recall: %.3f\tTest F-measure: %.3f" % (num_run,test_acc_th[num_run],test_hamming_loss_th[num_run],test_prec_th[num_run],test_rec_th[num_run],test_fmeasure_th[num_run]) + "\n" # also output the results of each run.
        output_csv_test = output_csv_test + "\n" + str(num_run) + "," + str(test_hamming_loss_th[num_run]) + "," + str(test_acc_th[num_run]) + "," + str(test_prec_th[num_run]) + "," + str(test_rec_th[num_run]) + "," + str(test_fmeasure_th[num_run])
        
        print("--- testing of fold %s took %s seconds ---" % (num_run, time.time() - start_time_test))
        
        prediction_str = ""
        # output final predictions for qualitative analysis
        if FLAGS.report_rand_pred == True:
            prediction_str = display_for_qualitative_evaluation(model, k_num_doc, mat_train, trainY, corpus_test, testX, testY, vocabulary_index2word, vocabulary_index2word_label, hamming_q=FLAGS.ave_labels_per_doc)
        # update the num_run
        num_run = num_run + 1
    
    print('\n--Final Results--\n')
    #print('C', FLAGS.C, 'gamma', FLAGS.gamma)
    
    # report min, max, std, average for the validation results
    min_valid_acc_th = min(valid_acc_th)
    min_valid_prec_th = min(valid_prec_th)
    min_valid_rec_th = min(valid_rec_th)
    min_valid_fmeasure_th = min(valid_fmeasure_th)
    min_valid_hamming_loss_th = min(valid_hamming_loss_th)
    
    max_valid_acc_th = max(valid_acc_th)
    max_valid_prec_th = max(valid_prec_th)
    max_valid_rec_th = max(valid_rec_th)
    max_valid_fmeasure_th = max(valid_fmeasure_th)
    max_valid_hamming_loss_th = max(valid_hamming_loss_th)
    
    if FLAGS.kfold != -1:
        std_valid_acc_th = statistics.stdev(valid_acc_th) # to change
        std_valid_prec_th = statistics.stdev(valid_prec_th)
        std_valid_rec_th = statistics.stdev(valid_rec_th)
        std_valid_fmeasure_th = statistics.stdev(valid_fmeasure_th)
        std_valid_hamming_loss_th = statistics.stdev(valid_hamming_loss_th)
    
    final_valid_acc_th = sum(valid_acc_th)/num_runs
    final_valid_prec_th = sum(valid_prec_th)/num_runs
    final_valid_rec_th = sum(valid_rec_th)/num_runs
    final_valid_fmeasure_th = sum(valid_fmeasure_th)/num_runs
    final_valid_hamming_loss_th = sum(valid_hamming_loss_th)/num_runs
    
    print("LDA==>Final Validation results Validation Accuracy: %.3f ± %.3f (%.3f - %.3f)\tValidation Hamming Loss: %.3f ± %.3f (%.3f - %.3f)\tValidation Precision: %.3f ± %.3f (%.3f - %.3f)\tValidation Recall: %.3f ± %.3f (%.3f - %.3f)\tValidation F-measure: %.3f ± %.3f (%.3f - %.3f)" % (final_valid_acc_th,std_valid_acc_th,min_valid_acc_th,max_valid_acc_th,final_valid_hamming_loss_th,std_valid_hamming_loss_th,min_valid_hamming_loss_th,max_valid_hamming_loss_th,final_valid_prec_th,std_valid_prec_th,min_valid_prec_th,max_valid_prec_th,final_valid_rec_th,std_valid_rec_th,min_valid_rec_th,max_valid_rec_th,final_valid_fmeasure_th,std_valid_fmeasure_th,min_valid_fmeasure_th,max_valid_fmeasure_th))
    #output the result to a file
    output_valid = output_valid + "\n" + "LDA==>Final Validation results Validation Accuracy: %.3f ± %.3f (%.3f - %.3f)\tValidation Hamming Loss: %.3f ± %.3f (%.3f - %.3f)\tValidation Precision: %.3f ± %.3f (%.3f - %.3f)\tValidation Recall: %.3f ± %.3f (%.3f - %.3f)\tValidation F-measure: %.3f ± %.3f (%.3f - %.3f)" % (final_valid_acc_th,std_valid_acc_th,min_valid_acc_th,max_valid_acc_th,final_valid_hamming_loss_th,std_valid_hamming_loss_th,min_valid_hamming_loss_th,max_valid_hamming_loss_th,final_valid_prec_th,std_valid_prec_th,min_valid_prec_th,max_valid_prec_th,final_valid_rec_th,std_valid_rec_th,min_valid_rec_th,max_valid_rec_th,final_valid_fmeasure_th,std_valid_fmeasure_th,min_valid_fmeasure_th,max_valid_fmeasure_th) + "\n"
    output_csv_valid = output_csv_valid + "\n" + "average" + "," + str(round(final_valid_hamming_loss_th,3)) + "±" + str(round(std_valid_hamming_loss_th,3)) + "," + str(round(final_valid_acc_th,3)) + "±" + str(round(std_valid_acc_th,3)) + "," + str(round(final_valid_prec_th,3)) + "±" + str(round(std_valid_prec_th,3)) + "," + str(round(final_valid_rec_th,3)) + "±" + str(round(std_valid_rec_th,3)) + "," + str(round(final_valid_fmeasure_th,3)) + "±" + str(round(std_valid_fmeasure_th,3))
    
    # report min, max, std, average for the testing results
    min_test_acc_th = min(test_acc_th)
    min_test_prec_th = min(test_prec_th)
    min_test_rec_th = min(test_rec_th)
    min_test_fmeasure_th = min(test_fmeasure_th)
    min_test_hamming_loss_th = min(test_hamming_loss_th)
    
    max_test_acc_th = max(test_acc_th)
    max_test_prec_th = max(test_prec_th)
    max_test_rec_th = max(test_rec_th)
    max_test_fmeasure_th = max(test_fmeasure_th)
    max_test_hamming_loss_th = max(test_hamming_loss_th)
    
    if FLAGS.kfold != -1:
        std_test_acc_th = statistics.stdev(test_acc_th) # to change
        std_test_prec_th = statistics.stdev(test_prec_th)
        std_test_rec_th = statistics.stdev(test_rec_th)
        std_test_fmeasure_th = statistics.stdev(test_fmeasure_th)
        std_test_hamming_loss_th = statistics.stdev(test_hamming_loss_th)
    
    final_test_acc_th = sum(test_acc_th)/num_runs
    final_test_prec_th = sum(test_prec_th)/num_runs
    final_test_rec_th = sum(test_rec_th)/num_runs
    final_test_fmeasure_th = sum(test_fmeasure_th)/num_runs
    final_test_hamming_loss_th = sum(test_hamming_loss_th)/num_runs
    
    print("LDA==>Final Test results Test Accuracy: %.3f ± %.3f (%.3f - %.3f)\tTest Hamming Loss: %.3f ± %.3f (%.3f - %.3f)\tTest Precision: %.3f ± %.3f (%.3f - %.3f)\tTest Recall: %.3f ± %.3f (%.3f - %.3f)\tTest F-measure: %.3f ± %.3f (%.3f - %.3f)" % (final_test_acc_th,std_test_acc_th,min_test_acc_th,max_test_acc_th,final_test_hamming_loss_th,std_test_hamming_loss_th,min_test_hamming_loss_th,max_test_hamming_loss_th,final_test_prec_th,std_test_prec_th,min_test_prec_th,max_test_prec_th,final_test_rec_th,std_test_rec_th,min_test_rec_th,max_test_rec_th,final_test_fmeasure_th,std_test_fmeasure_th,min_test_fmeasure_th,max_test_fmeasure_th))
    #output the result to a file
    output_test = output_test + "\n" + "LDA==>Final Test results Test Accuracy: %.3f ± %.3f (%.3f - %.3f)\tTest Hamming Loss: %.3f ± %.3f (%.3f - %.3f)\tTest Precision: %.3f ± %.3f (%.3f - %.3f)\tTest Recall: %.3f ± %.3f (%.3f - %.3f)\tTest F-measure: %.3f ± %.3f (%.3f - %.3f)" % (final_test_acc_th,std_test_acc_th,min_test_acc_th,max_test_acc_th,final_test_hamming_loss_th,std_test_hamming_loss_th,min_test_hamming_loss_th,max_test_hamming_loss_th,final_test_prec_th,std_test_prec_th,min_test_prec_th,max_test_prec_th,final_test_rec_th,std_test_rec_th,min_test_rec_th,max_test_rec_th,final_test_fmeasure_th,std_test_fmeasure_th,min_test_fmeasure_th,max_test_fmeasure_th) + "\n"
    output_csv_test = output_csv_test + "\n" + "average" + "," + str(round(final_test_hamming_loss_th,3)) + "±" + str(round(std_test_hamming_loss_th,3)) + "," + str(round(final_test_acc_th,3)) + "±" + str(round(std_test_acc_th,3)) + "," + str(round(final_test_prec_th,3)) + "±" + str(round(std_test_prec_th,3)) + "," + str(round(final_test_rec_th,3)) + "±" + str(round(std_test_rec_th,3)) + "," + str(round(final_test_fmeasure_th,3)) + "±" + str(round(std_test_fmeasure_th,3))
    
    setting = "dataset:" + str(FLAGS.dataset) + "\nT: " + str(FLAGS.num_topics) + "\nk: " + str(FLAGS.k_num_doc) + ' \ni: ' + str(FLAGS.iterations)
    print("--- The whole program took %s seconds ---" % (time.time() - start_time))
    time_used = "--- The whole program took %s seconds ---" % (time.time() - start_time)
    if FLAGS.kfold != -1:
        print("--- The average training took %s ± %s seconds ---" % (sum(time_train)/num_runs,statistics.stdev(time_train)))
        average_time_train = "--- The average training took %s ± %s seconds ---" % (sum(time_train)/num_runs,statistics.stdev(time_train))
    else:
        print("--- The average training took %s ± %s seconds ---" % (sum(time_train)/num_runs,0))
        average_time_train = "--- The average training took %s ± %s seconds ---" % (sum(time_train)/num_runs,0)

    # output setting configuration, results, prediction and time used
    output_to_file('lda ' + str(FLAGS.dataset) + " T" + str(FLAGS.num_topics) + ' k' + str(FLAGS.k_num_doc) + ' i' + str(FLAGS.iterations) + ' gp_id' + str(FLAGS.marking_id) + '.txt',setting + '\n' + output_valid + '\n' + output_test + '\n' + prediction_str + '\n' + time_used + '\n' + average_time_train)
    # output structured evaluation results
    output_to_file('lda ' + str(FLAGS.dataset) + " T" + str(FLAGS.num_topics) + ' k' + str(FLAGS.k_num_doc) + ' i' + str(FLAGS.iterations) + ' gp_id' + str(FLAGS.marking_id) + ' valid.csv',output_csv_valid)
    output_to_file('lda ' + str(FLAGS.dataset) + " T" + str(FLAGS.num_topics) + ' k' + str(FLAGS.k_num_doc) + ' i' + str(FLAGS.iterations) + ' gp_id' + str(FLAGS.marking_id) + ' test.csv',output_csv_test)
    
def generateLDAdocFromIndex(dataX, vocabulary_index2word, remove_pad_id=False, remove_dot=False):
    docs=[]
    for doc in dataX:
        words=[]
        for index in doc:
            if not (remove_pad_id and index == 0):
                if not (remove_dot and vocabulary_index2word[index] == '.'):
                        words.append(vocabulary_index2word[index])           
            # if remove_pad_id:
                # if index != 0:
                    # if not (remove_dot and vocabulary_index2word[index] == '.'):
                        # words.append(vocabulary_index2word[index])                    
            # else:
                # if not (remove_dot and vocabulary_index2word[index] == '.'):
                    # words.append(vocabulary_index2word[index])
        docs.append(words)
    return docs

#get top similar docs (indexes) from similarity vector
def get_doc_ind_from_vec(vec,k_num_doc=1):
    index_list=np.argsort(vec)[-k_num_doc:]
    index_list=index_list[::-1]
    return index_list

def get_labels_from_docs(index_list,dataY):
    labels = []
    for index in index_list:
        #print(labels)
        #print(get_true_label_ind(dataY[index]))
        labels = union(labels,get_true_label_ind(dataY[index]))
    return labels
    
# https://www.geeksforgeeks.org/python-union-two-lists/
# Python program to illustrate union 
# Without repetition  
def union(lst1, lst2): 
    final_list = list(set(lst1) | set(lst2)) 
    return final_list

def display_results(index_list,index2word):
    label_list=[]
    for index in index_list:
        if index!=0: # this ensures that the padded values not being displayed.
            label=index2word[index]
            label_list.append(label)
    return label_list

# get a list of true label indexes from a multihot vector
def get_true_label_ind(multihot):
    label_nozero=[]
    #print("labels:",labels)
    multihot=list(multihot)
    for index,label in enumerate(multihot):
        if label>0:
            label_nozero.append(index)
    return label_nozero
    
def output_to_file(file_name,str):
    with open(file_name, 'w', encoding="utf-8-sig") as f_output:
        f_output.write(str + '\n')
    
def do_eval_lda(modelToEval, k_num_doc, mat_train, trainY, corpus_eval, evalY, vocabulary_index2word_label, hamming_q=FLAGS.ave_labels_per_doc):
    # get eval-train document similarity matrix
    #mat_train = np.array(modelToEval[corpus]) #https://stackoverflow.com/questions/21322564/numpy-list-of-1d-arrays-to-2d-array
    #get_document_topics(self, bow, minimum_probability=None, minimum_phi_value=None, per_word_topics=False):
    #mat_train = np.array(modelToEval.get_document_topics(corpus,minimum_probability=0.0))
    #print(len(modelToEval[corpus[0]]))
    #print(len(modelToEval[corpus[1]]))
    #print(len(modelToEval[corpus[2]]))
    #print(mat_train.shape)
    #mat_train = mat_train[:,:,1] #https://stackoverflow.com/questions/37152031/numpy-remove-a-dimension-from-np-array
    mat_eval = np.array(modelToEval.get_document_topics(corpus_eval,minimum_probability=0.0))
    #print(mat_eval.shape)
    mat_eval = mat_eval[:,:,1]
    mat_sim_v_tr = cosine_similarity(mat_eval,mat_train) # a matrix (n_valid,n_train)
    
    y_true = np.asarray(evalY)
    acc, prec, rec, hamming_loss = 0.0, 0.0, 0.0, 0.0
    for i in range(len(mat_sim_v_tr)):
        doc_ind_list = get_doc_ind_from_vec(mat_sim_v_tr[i],k_num_doc=k_num_doc)
        #print(doc_ind_list)
        label_predicted = get_labels_from_docs(doc_ind_list,trainY)
        #print(label_ind_list)
        #label_predicted = display_results(label_ind_list,vocabulary_index2word_label)
        #print(label_predicted)
        
        curr_acc = calculate_accuracy(label_predicted,y_true[i])
        acc = acc + curr_acc
        curr_prec, curr_rec = calculate_precision_recall(label_predicted,y_true[i])
        prec = prec + curr_prec
        rec = rec + curr_rec
        curr_hl = calculate_hamming_loss(label_predicted,y_true[i])
        hamming_loss = hamming_loss + curr_hl
    acc = acc/float(len(mat_sim_v_tr))
    prec = prec/float(len(mat_sim_v_tr))
    rec = rec/float(len(mat_sim_v_tr))
    hamming_loss = hamming_loss/float(len(mat_sim_v_tr))/FLAGS.ave_labels_per_doc
    if prec+rec != 0:
        f_measure = 2*prec*rec/(prec+rec)
    else:
        f_measure = 0
    return acc,prec,rec,f_measure,hamming_loss

# this also needs evalX
def display_for_qualitative_evaluation(modelToEval, k_num_doc, mat_train, trainY, corpus_eval, evalX, evalY, vocabulary_index2word, vocabulary_index2word_label, hamming_q=FLAGS.ave_labels_per_doc):
    prediction_str=""
    #generate the doc indexes same as for the deep learning models.
    number_examples=len(evalY)
    rn_dict={}
    rn.seed(1) # set the seed to produce same documents for prediction
    batch_size=128
    for i in range(0,500):
        batch_chosen=rn.randint(0,number_examples//batch_size)
        x_chosen=rn.randint(0,batch_size)
        #rn_dict[(batch_chosen*batch_size,x_chosen)]=1
        rn_dict[batch_chosen*batch_size+x_chosen]=1
        
    # get eval-train document similarity matrix
    #mat_train = np.array(modelToEval[corpus]) #https://stackoverflow.com/questions/21322564/numpy-list-of-1d-arrays-to-2d-array
    #mat_train = mat_train[:,:,1] #https://stackoverflow.com/questions/37152031/numpy-remove-a-dimension-from-np-array
    #mat_eval = np.array(modelToEval[corpus_eval])
    mat_eval = np.array(modelToEval.get_document_topics(corpus_eval,minimum_probability=0.0))
    mat_eval = mat_eval[:,:,1]
    mat_sim_v_tr = cosine_similarity(mat_eval,mat_train) # a matrix (n_valid,n_train)
    
    y_true = np.asarray(evalY)    
    for i in range(len(mat_sim_v_tr)):
        doc_ind_list = get_doc_ind_from_vec(mat_sim_v_tr[i],k_num_doc=k_num_doc)
        #print(doc_ind_list)
        label_predicted = get_labels_from_docs(doc_ind_list,trainY)
        if rn_dict.get(i) == 1:
            doc = 'doc: ' + ' '.join(display_results(evalX[i],vocabulary_index2word))
            pred = 'prediction-lda: ' + ' '.join(display_results(label_predicted,vocabulary_index2word_label))
            get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
            label = 'labels: ' + ' '.join(display_results(get_indexes(1,evalY[i]),vocabulary_index2word_label))
            prediction_str = prediction_str + '\n' + doc + '\n' + pred + '\n' + label + '\n'
    
    return prediction_str    
    
def calculate_accuracy(labels_predicted,labels): # this should be same as the recall value
    # turn the multihot representation to a list of true labels
    label_nozero=[]
    #print("labels:",labels)
    labels=list(labels)
    for index,label in enumerate(labels):
        if label>0:
            label_nozero.append(index)
    #if eval_counter<2:
    #    print("labels_predicted:",labels_predicted," ;labels_nozero:",label_nozero)
    overlapping = 0
    label_dict = {x: x for x in label_nozero} # create a dictionary of labels for the true labels
    union = len(label_dict)
    for label_predict in labels_predicted:
        flag = label_dict.get(label_predict, None)
        if flag is not None:
            overlapping = overlapping + 1
        else:
            union = union + 1        
    return overlapping / union

def calculate_precision_recall(labels_predicted, labels):
    label_nozero=[]
    #print("labels:",labels)
    labels=list(labels)
    for index,label in enumerate(labels):
        if label>0:
            label_nozero.append(index)
    #if eval_counter<2:
    #    print("labels_predicted:",labels_predicted," ;labels_nozero:",label_nozero)
    count = 0
    label_dict = {x: x for x in label_nozero}
    for label_predict in labels_predicted:
        flag = label_dict.get(label_predict, None)
        if flag is not None:
            count = count + 1
    if (len(labels_predicted)==0): # if nothing predicted, then set the precision as 0.
        precision=0
    else: 
        precision = count / len(labels_predicted)
    recall = count / len(label_nozero)
    #fmeasure = 2*precision*recall/(precision+recall)
    #print(count, len(label_nozero))
    return precision, recall
   
# calculate the symmetric_difference
def calculate_hamming_loss(labels_predicted, labels):
    label_nozero=[]
    #print("labels:",labels)
    labels=list(labels)
    for index,label in enumerate(labels):
        if label>0:
            label_nozero.append(index)
    count = 0
    label_dict = {x: x for x in label_nozero} # get the true labels

    for label_predict in labels_predicted:
        flag = label_dict.get(label_predict, None)
        if flag is not None:
            count = count + 1 # get the number of overlapping labels
    
    return len(label_dict)+len(labels_predicted)-2*count
    
if __name__ == "__main__":
    tf.app.run()
