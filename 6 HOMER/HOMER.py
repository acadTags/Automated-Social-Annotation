#this program is based on the meka wrapper from scikit-multilearn
import tensorflow as tf
import numpy as np
import time

from data_util import load_data_multilabel_new, load_data_multilabel_new_k_fold, create_voabulary, \
    create_voabulary_label
from tflearn.data_utils import to_categorical, pad_sequences
import os
import sys
import word2vec
import pickle
import random as rn
import statistics
import warnings

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn import metrics

from scipy import sparse
#from skmultilearn.problem_transform import BinaryRelevance
#from skmultilearn.ext import download_meka
from skmultilearn.ext import Meka

# start time
start_time = time.time()

# only using CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# notification/warning settings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #block tensorflow logs
warnings.filterwarnings("ignore", category=UserWarning)  # block sklearn UserWarning

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("dataset", "bibsonomy-clean",
                           "dataset to chose")  # two options: "bibsonomy-clean" and "zhihu-sample"
# tf.app.flags.DEFINE_string("dataset","zhihu-sample","dataset to chose") # two options: "bibsonomy-clean" and "zhihu-sample"

tf.app.flags.DEFINE_integer("sequence_length", 300, "max sentence length")
tf.app.flags.DEFINE_integer("embed_size", 100, "embedding size")

# for simulating missing labels
tf.app.flags.DEFINE_float("keep_label_percent", 1,
                          "the percentage of labels in each instance of the training data to be randomly reserved, the rest labels are dropped to simulate the missing label scenario.")

# for both tuning and final testing
tf.app.flags.DEFINE_string("training_data_path_bib", "../datasets/bibsonomy_preprocessed_merged_final.txt",
                           "path of traning data.")  # for bibsonomy dataset
tf.app.flags.DEFINE_string("training_data_path_zhihu", "../datasets/question_train_set_cleaned_150000.txt",
                           "path of traning data.")  # for zhihu dataset
tf.app.flags.DEFINE_string("training_data_path_cua", "../datasets/citeulike_a_cleaned_th10.txt",
                           "path of traning data.")  # for cua dataset
tf.app.flags.DEFINE_string("training_data_path_cut", "../datasets/citeulike_t_cleaned_th10.txt",
                           "path of traning data.")  # for cut dataset

tf.app.flags.DEFINE_string("word2vec_model_path_bib", "../embeddings/word-bib.bin-100",
                           "word2vec's vocabulary and vectors for inputs")
tf.app.flags.DEFINE_string("word2vec_model_path_zhihu", "../embeddings/word150000.bin-100",
                           "word2vec's vocabulary and vectors")
tf.app.flags.DEFINE_string("word2vec_model_path_cua", "../embeddings/word-citeulike-a.bin-100",
                           "word2vec's vocabulary and vectors")
tf.app.flags.DEFINE_string("word2vec_model_path_cut", "../embeddings/word-citeulike-t-th10.bin-100",
                           "word2vec's vocabulary and vectors")

tf.app.flags.DEFINE_float("valid_portion", 0.1,
                          "dev set or test set portion")  # this is only valid when kfold is -1, which means we hold out a fixed set for validation.
# if we set this as 0.1, then there will be 0.81 0.09 0.1 for train-valid-test split (same as the split of 10-fold cross-validation); if we set this as 0.111, then there will be 0.8 0.1 0.1 for train-valid-test split.
tf.app.flags.DEFINE_float("test_portion", 0.1, "held-out evaluation: test set portion")
tf.app.flags.DEFINE_integer("kfold", 10, "k-fold cross-validation")  # if k is -1, then not using kfold cross-validation

tf.app.flags.DEFINE_string("marking_id", "",
                           "an marking_id (or group_id) for better marking: will show in the output filenames")

tf.app.flags.DEFINE_boolean("multi_label_flag", True, "use multi label or single label.")
tf.app.flags.DEFINE_boolean("report_rand_pred", True, "report prediction for qualitative analysis")
tf.app.flags.DEFINE_float("ave_labels_per_doc", 11.59, "average labels per document for bibsonomy dataset")
# tf.app.flags.DEFINE_integer("topk",5,"using top-k predicted labels for evaluation")

tf.app.flags.DEFINE_string("mode", "linear", "SVM mode: linear or rbf [unused for HOMER]")
tf.app.flags.DEFINE_float("C", 0, "parameter C in svm rbf kernel")
tf.app.flags.DEFINE_float("gamma", 0, "parameter gamma in svm rbf kernel")

tf.app.flags.DEFINE_integer("num_clusters", 0, "number of clusters for balanced k-means in the HOMER meta classifier")


# 0. configuration
# todo: add command-line argument

def main(_):
    # os.environ['CUDA_VISIBLE_DEVICES'] = ''

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
    vocabulary_word2index, vocabulary_index2word = create_voabulary(word2vec_model_path,
                                                                    name_scope=FLAGS.dataset + "-svm")  # simple='simple'
    vocabulary_word2index_label, vocabulary_index2word_label = create_voabulary_label(voabulary_label=traning_data_path,
                                                                                      name_scope=FLAGS.dataset + "-svm")
    num_classes = len(vocabulary_word2index_label)
    print(vocabulary_index2word_label[0], vocabulary_index2word_label[1])

    vocab_size = len(vocabulary_word2index)
    print("vocab_size:", vocab_size)

    # choosing whether to use k-fold cross-validation or hold-out validation
    if FLAGS.kfold == -1:  # hold-out
        train, valid, test = load_data_multilabel_new(vocabulary_word2index, vocabulary_word2index_label,
                                                      keep_label_percent=FLAGS.keep_label_percent,
                                                      valid_portion=FLAGS.valid_portion,
                                                      test_portion=FLAGS.test_portion,
                                                      multi_label_flag=FLAGS.multi_label_flag,
                                                      traning_data_path=traning_data_path)
        # here train, test are tuples; turn train into trainlist.
        trainlist, validlist, testlist = list(), list(), list()
        trainlist.append(train)
        validlist.append(valid)
        testlist.append(test)
    else:  # k-fold
        trainlist, validlist, testlist = load_data_multilabel_new_k_fold(vocabulary_word2index,
                                                                         vocabulary_word2index_label,
                                                                         keep_label_percent=FLAGS.keep_label_percent,
                                                                         kfold=FLAGS.kfold,
                                                                         test_portion=FLAGS.test_portion,
                                                                         multi_label_flag=FLAGS.multi_label_flag,
                                                                         traning_data_path=traning_data_path)
        # here trainlist, testlist are list of tuples.
    # get and pad testing data: there is only one testing data, but kfold training and validation data
    assert len(testlist) == 1
    testX, testY = testlist[0]
    testX = pad_sequences(testX, maxlen=FLAGS.sequence_length, value=0.)  # padding to max length

    # 2. get word_embedding matrix: shape (21425,100)
    word2vec_model = word2vec.load(word2vec_model_path, kind='bin')
    word2vec_dict = {}
    for word, vector in zip(word2vec_model.vocab, word2vec_model.vectors):
        word2vec_dict[word] = vector
    word_embedding_2dlist = [
                                []] * vocab_size  # create an empty word_embedding list: which is a list of list, i.e. a list of word, where each word is a list of values as an embedding vector.
    word_embedding_2dlist[0] = np.zeros(FLAGS.embed_size)  # assign empty for first word:'PAD'
    bound = np.sqrt(6.0) / np.sqrt(vocab_size)  # bound for random variables.
    count_exist = 0;
    count_not_exist = 0
    for i in range(1, vocab_size):  # loop each word
        word = vocabulary_index2word[i]  # get a word
        embedding = None
        try:
            embedding = word2vec_dict[word]  # try to get vector:it is an array.
        except Exception:
            embedding = None
        if embedding is not None:  # the 'word' exist a embedding
            word_embedding_2dlist[i] = embedding;
            count_exist = count_exist + 1  # assign array to this word.
        else:  # no embedding for this word
            word_embedding_2dlist[i] = np.random.uniform(-bound, bound, FLAGS.embed_size);
            count_not_exist = count_not_exist + 1  # init a random value for the word.
    word_embedding_final = np.array(word_embedding_2dlist)  # covert to 2d array.
    print('embedding per word:', word_embedding_final)
    print('embedding per word, shape:', word_embedding_final.shape)

    # 3. transform trainlist to the format. x_train, x_test: training and test feature matrices of size (n_samples, n_features)
    # print(len(trainlist))
    # trainX,trainY = trainlist[0]
    # trainX = pad_sequences(trainX, maxlen=FLAGS.sequence_length, value=0.)
    # print(len(trainX))
    # print(len(trainX[0]))
    # print(trainX[0])
    # print(len(trainY))
    # print(len(trainY[0]))
    # print(trainY[0])
    # print(np.asarray(trainY).shape)

    num_runs = len(trainlist)
    # validation results variables
    valid_acc_th, valid_prec_th, valid_rec_th, valid_fmeasure_th, valid_hamming_loss_th = [0] * num_runs, [
        0] * num_runs, [0] * num_runs, [0] * num_runs, [0] * num_runs  # initialise the result lists
    final_valid_acc_th, final_valid_prec_th, final_valid_rec_th, final_valid_fmeasure_th, final_valid_hamming_loss_th = 0.0, 0.0, 0.0, 0.0, 0.0
    min_valid_acc_th, min_valid_prec_th, min_valid_rec_th, min_valid_fmeasure_th, min_valid_hamming_loss_th = 0.0, 0.0, 0.0, 0.0, 0.0
    max_valid_acc_th, max_valid_prec_th, max_valid_rec_th, max_valid_fmeasure_th, max_valid_hamming_loss_th = 0.0, 0.0, 0.0, 0.0, 0.0
    std_valid_acc_th, std_valid_prec_th, std_valid_rec_th, std_valid_fmeasure_th, std_valid_hamming_loss_th = 0.0, 0.0, 0.0, 0.0, 0.0
    # testing results variables
    test_acc_th, test_prec_th, test_rec_th, test_fmeasure_th, test_hamming_loss_th = [0] * num_runs, [0] * num_runs, [
        0] * num_runs, [0] * num_runs, [0] * num_runs  # initialise the testing result lists
    final_test_acc_th, final_test_prec_th, final_test_rec_th, final_test_fmeasure_th, final_test_hamming_loss_th = 0.0, 0.0, 0.0, 0.0, 0.0
    min_test_acc_th, min_test_prec_th, min_test_rec_th, min_test_fmeasure_th, min_test_hamming_loss_th = 0.0, 0.0, 0.0, 0.0, 0.0
    max_test_acc_th, max_test_prec_th, max_test_rec_th, max_test_fmeasure_th, max_test_hamming_loss_th = 0.0, 0.0, 0.0, 0.0, 0.0
    std_test_acc_th, std_test_prec_th, std_test_rec_th, std_test_fmeasure_th, std_test_hamming_loss_th = 0.0, 0.0, 0.0, 0.0, 0.0
    # output variables
    output_valid = ""
    output_test = ""
    output_csv_valid = "fold,hamming_loss,acc,prec,rec,f1"
    output_csv_test = "fold,hamming_loss,acc,prec,rec,f1"

    time_train = [0] * num_runs  # get time spent in training
    num_run = 0
    testX_embedded = get_embedded_words(testX, word_embedding_final, vocab_size)
    print('testX_embedded:', testX_embedded)
    print('testX_embedded:', testX_embedded.shape)

    for trainfold in trainlist:
        # get training and validation data
        trainX, trainY = trainfold
        trainX = pad_sequences(trainX, maxlen=FLAGS.sequence_length, value=0.)
        trainX_embedded = get_embedded_words(trainX, word_embedding_final, vocab_size)
        print('trainX_embedded:', trainX_embedded)
        print('trainX_embedded:', trainX_embedded.shape)

        #print('trainX_embedded_for_debugging:',trainX_embedded[0:1000].shape) # for quick debugging
        #trainX_embedded = trainX_embedded[0:1000] # for quick debugging
        #trainY = trainY[0:1000] # for quick debugging

        validX, validY = validlist[num_run]
        validX = pad_sequences(validX, maxlen=FLAGS.sequence_length, value=0.)
        validX_embedded = get_embedded_words(validX, word_embedding_final, vocab_size)
        print('validX_embedded:', validX_embedded)
        print('validX_embedded:', validX_embedded.shape)

    # training
        start_time_train = time.time()
        print('start training fold', str(num_run))

        # check trainY and remove labels that are False for all training instances
        trainY_int = np.asarray(trainY).astype(int)
        one_class_label_list = list()  # the list of labels that are not associated with any training instances.
        # print(trainY_int.shape)
        # print(sum(trainY_int[:,2]))
        for k in range(num_classes):
            if sum(trainY_int[:, k]) == 0:
                # print(k)
                one_class_label_list.append(k)
        # to delete the labels not associated to any labels in the training data
        trainY_int_pruned = np.delete(trainY_int, one_class_label_list, 1)
        print(trainY_int_pruned.shape)

        model = train_HOMER(trainX_embedded, trainY_int_pruned)
        print('num_run', str(num_run), 'train done.')

        time_train[num_run] = time.time() - start_time_train
        print("--- training of fold %s took %s seconds ---" % (num_run, time_train[num_run]))

        # evaluate on training data
        acc, prec, rec, f_measure, hamming_loss = do_eval(model, trainX_embedded, np.asarray(trainY),
                                                          hamming_q=FLAGS.ave_labels_per_doc, one_class_label_list=one_class_label_list)
        # print('training:', acc, prec, rec, f_measure, hamming_loss)
        # pp = model.predict_proba(trainX_embedded)
        # print('pp',pp)
        # print('pp:',pp.shape)
        # print('pp_sum',np.sum(pp,0))
        # print('pp_sum',np.sum(pp,1))

        # evaluate on validation data
        valid_acc_th[num_run], valid_prec_th[num_run], valid_rec_th[num_run], valid_fmeasure_th[num_run], \
        valid_hamming_loss_th[num_run] = do_eval(model, validX_embedded, validY, hamming_q=FLAGS.ave_labels_per_doc, one_class_label_list=one_class_label_list)
        # print('validation:', acc, prec, rec, f_measure, hamming_loss)
        print(
                    "HOMER==>Run %d Validation Accuracy: %.3f\tValidation Hamming Loss: %.3f\tValidation Precision: %.3f\tValidation Recall: %.3f\tValidation F-measure: %.3f" % (
            num_run, valid_acc_th[num_run], valid_hamming_loss_th[num_run], valid_prec_th[num_run],
            valid_rec_th[num_run], valid_fmeasure_th[num_run]))
        output_valid = output_valid + "\n" + "HOMER==>Run %d Validation Accuracy: %.3f\tValidation Hamming Loss: %.3f\tValidation Precision: %.3f\tValidation Recall: %.3f\tValidation F-measure: %.3f" % (
        num_run, valid_acc_th[num_run], valid_hamming_loss_th[num_run], valid_prec_th[num_run], valid_rec_th[num_run],
        valid_fmeasure_th[num_run]) + "\n"  # also output the results of each run.
        output_csv_valid = output_csv_valid + "\n" + str(num_run) + "," + str(
            valid_hamming_loss_th[num_run]) + "," + str(valid_acc_th[num_run]) + "," + str(
            valid_prec_th[num_run]) + "," + str(valid_rec_th[num_run]) + "," + str(valid_fmeasure_th[num_run])

        start_time_test = time.time()
        # evaluate on testing data
        test_acc_th[num_run], test_prec_th[num_run], test_rec_th[num_run], test_fmeasure_th[num_run], \
        test_hamming_loss_th[num_run] = do_eval(model, testX_embedded, testY, hamming_q=FLAGS.ave_labels_per_doc, one_class_label_list=one_class_label_list)
        # print('testing:', acc, prec, rec, f_measure, hamming_loss)
        print(
                    "HOMER==>Run %d Test Accuracy: %.3f\tTest Hamming Loss: %.3f\tTest Precision: %.3f\tTest Recall: %.3f\tTest F-measure: %.3f" % (
            num_run, test_acc_th[num_run], test_hamming_loss_th[num_run], test_prec_th[num_run], test_rec_th[num_run],
            test_fmeasure_th[num_run]))
        output_test = output_test + "\n" + "HOMER==>Run %d Test Accuracy: %.3f\tTest Hamming Loss: %.3f\tTest Precision: %.3f\tTest Recall: %.3f\tTest F-measure: %.3f" % (
        num_run, test_acc_th[num_run], test_hamming_loss_th[num_run], test_prec_th[num_run], test_rec_th[num_run],
        test_fmeasure_th[num_run]) + "\n"  # also output the results of each run.
        output_csv_test = output_csv_test + "\n" + str(num_run) + "," + str(test_hamming_loss_th[num_run]) + "," + str(
            test_acc_th[num_run]) + "," + str(test_prec_th[num_run]) + "," + str(test_rec_th[num_run]) + "," + str(
            test_fmeasure_th[num_run])

        print("--- testing of fold %s took %s seconds ---" % (num_run, time.time() - start_time_test))

        prediction_str = ""
        # output final predictions for qualitative analysis
        if FLAGS.report_rand_pred == True:
            prediction_str = display_for_qualitative_evaluation(model, testX_embedded, testX, testY,
                                                                vocabulary_index2word, vocabulary_index2word_label, one_class_label_list=one_class_label_list)
        # update the num_run
        num_run = num_run + 1

    print('\n--Final Results--\n')
    # print('C', FLAGS.C, 'gamma', FLAGS.gamma)

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
        std_valid_acc_th = statistics.stdev(valid_acc_th)  # to change
        std_valid_prec_th = statistics.stdev(valid_prec_th)
        std_valid_rec_th = statistics.stdev(valid_rec_th)
        std_valid_fmeasure_th = statistics.stdev(valid_fmeasure_th)
        std_valid_hamming_loss_th = statistics.stdev(valid_hamming_loss_th)

    final_valid_acc_th = sum(valid_acc_th) / num_runs
    final_valid_prec_th = sum(valid_prec_th) / num_runs
    final_valid_rec_th = sum(valid_rec_th) / num_runs
    final_valid_fmeasure_th = sum(valid_fmeasure_th) / num_runs
    final_valid_hamming_loss_th = sum(valid_hamming_loss_th) / num_runs

    print(
                "HOMER==>Final Validation results Validation Accuracy: %.3f ± %.3f (%.3f - %.3f)\tValidation Hamming Loss: %.3f ± %.3f (%.3f - %.3f)\tValidation Precision: %.3f ± %.3f (%.3f - %.3f)\tValidation Recall: %.3f ± %.3f (%.3f - %.3f)\tValidation F-measure: %.3f ± %.3f (%.3f - %.3f)" % (
        final_valid_acc_th, std_valid_acc_th, min_valid_acc_th, max_valid_acc_th, final_valid_hamming_loss_th,
        std_valid_hamming_loss_th, min_valid_hamming_loss_th, max_valid_hamming_loss_th, final_valid_prec_th,
        std_valid_prec_th, min_valid_prec_th, max_valid_prec_th, final_valid_rec_th, std_valid_rec_th, min_valid_rec_th,
        max_valid_rec_th, final_valid_fmeasure_th, std_valid_fmeasure_th, min_valid_fmeasure_th, max_valid_fmeasure_th))
    # output the result to a file
    output_valid = output_valid + "\n" + "HOMER==>Final Validation results Validation Accuracy: %.3f ± %.3f (%.3f - %.3f)\tValidation Hamming Loss: %.3f ± %.3f (%.3f - %.3f)\tValidation Precision: %.3f ± %.3f (%.3f - %.3f)\tValidation Recall: %.3f ± %.3f (%.3f - %.3f)\tValidation F-measure: %.3f ± %.3f (%.3f - %.3f)" % (
    final_valid_acc_th, std_valid_acc_th, min_valid_acc_th, max_valid_acc_th, final_valid_hamming_loss_th,
    std_valid_hamming_loss_th, min_valid_hamming_loss_th, max_valid_hamming_loss_th, final_valid_prec_th,
    std_valid_prec_th, min_valid_prec_th, max_valid_prec_th, final_valid_rec_th, std_valid_rec_th, min_valid_rec_th,
    max_valid_rec_th, final_valid_fmeasure_th, std_valid_fmeasure_th, min_valid_fmeasure_th,
    max_valid_fmeasure_th) + "\n"
    output_csv_valid = output_csv_valid + "\n" + "average" + "," + str(
        round(final_valid_hamming_loss_th, 3)) + "±" + str(round(std_valid_hamming_loss_th, 3)) + "," + str(
        round(final_valid_acc_th, 3)) + "±" + str(round(std_valid_acc_th, 3)) + "," + str(
        round(final_valid_prec_th, 3)) + "±" + str(round(std_valid_prec_th, 3)) + "," + str(
        round(final_valid_rec_th, 3)) + "±" + str(round(std_valid_rec_th, 3)) + "," + str(
        round(final_valid_fmeasure_th, 3)) + "±" + str(round(std_valid_fmeasure_th, 3))

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
        std_test_acc_th = statistics.stdev(test_acc_th)  # to change
        std_test_prec_th = statistics.stdev(test_prec_th)
        std_test_rec_th = statistics.stdev(test_rec_th)
        std_test_fmeasure_th = statistics.stdev(test_fmeasure_th)
        std_test_hamming_loss_th = statistics.stdev(test_hamming_loss_th)

    final_test_acc_th = sum(test_acc_th) / num_runs
    final_test_prec_th = sum(test_prec_th) / num_runs
    final_test_rec_th = sum(test_rec_th) / num_runs
    final_test_fmeasure_th = sum(test_fmeasure_th) / num_runs
    final_test_hamming_loss_th = sum(test_hamming_loss_th) / num_runs

    print(
                "HOMER==>Final Test results Test Accuracy: %.3f ± %.3f (%.3f - %.3f)\tTest Hamming Loss: %.3f ± %.3f (%.3f - %.3f)\tTest Precision: %.3f ± %.3f (%.3f - %.3f)\tTest Recall: %.3f ± %.3f (%.3f - %.3f)\tTest F-measure: %.3f ± %.3f (%.3f - %.3f)" % (
        final_test_acc_th, std_test_acc_th, min_test_acc_th, max_test_acc_th, final_test_hamming_loss_th,
        std_test_hamming_loss_th, min_test_hamming_loss_th, max_test_hamming_loss_th, final_test_prec_th,
        std_test_prec_th, min_test_prec_th, max_test_prec_th, final_test_rec_th, std_test_rec_th, min_test_rec_th,
        max_test_rec_th, final_test_fmeasure_th, std_test_fmeasure_th, min_test_fmeasure_th, max_test_fmeasure_th))
    # output the result to a file
    output_test = output_test + "\n" + "HOMER==>Final Test results Test Accuracy: %.3f ± %.3f (%.3f - %.3f)\tTest Hamming Loss: %.3f ± %.3f (%.3f - %.3f)\tTest Precision: %.3f ± %.3f (%.3f - %.3f)\tTest Recall: %.3f ± %.3f (%.3f - %.3f)\tTest F-measure: %.3f ± %.3f (%.3f - %.3f)" % (
    final_test_acc_th, std_test_acc_th, min_test_acc_th, max_test_acc_th, final_test_hamming_loss_th,
    std_test_hamming_loss_th, min_test_hamming_loss_th, max_test_hamming_loss_th, final_test_prec_th, std_test_prec_th,
    min_test_prec_th, max_test_prec_th, final_test_rec_th, std_test_rec_th, min_test_rec_th, max_test_rec_th,
    final_test_fmeasure_th, std_test_fmeasure_th, min_test_fmeasure_th, max_test_fmeasure_th) + "\n"
    output_csv_test = output_csv_test + "\n" + "average" + "," + str(round(final_test_hamming_loss_th, 3)) + "±" + str(
        round(std_test_hamming_loss_th, 3)) + "," + str(round(final_test_acc_th, 3)) + "±" + str(
        round(std_test_acc_th, 3)) + "," + str(round(final_test_prec_th, 3)) + "±" + str(
        round(std_test_prec_th, 3)) + "," + str(round(final_test_rec_th, 3)) + "±" + str(
        round(std_test_rec_th, 3)) + "," + str(round(final_test_fmeasure_th, 3)) + "±" + str(
        round(std_test_fmeasure_th, 3))

    setting = "dataset:" + str(FLAGS.dataset) + "\nC: " + str(FLAGS.C) + "\ngamma: " + str(FLAGS.gamma) + "\nnum_clusters: " + str(FLAGS.num_clusters)
    print("--- The whole program took %s seconds ---" % (time.time() - start_time))
    time_used = "--- The whole program took %s seconds ---" % (time.time() - start_time)
    if FLAGS.kfold != -1:
        print("--- The average training took %s ± %s seconds ---" % (
        sum(time_train) / num_runs, statistics.stdev(time_train)))
        average_time_train = "--- The average training took %s ± %s seconds ---" % (
        sum(time_train) / num_runs, statistics.stdev(time_train))
    else:
        print("--- The average training took %s ± %s seconds ---" % (sum(time_train) / num_runs, 0))
        average_time_train = "--- The average training took %s ± %s seconds ---" % (sum(time_train) / num_runs, 0)

    # output setting configuration, results, prediction and time used
    output_to_file('HOMER ' + str(FLAGS.dataset) + " C " + str(FLAGS.C) + ' gamma' + str(FLAGS.gamma) + ' num_clusters' + str(FLAGS.num_clusters) + ' gp_id' + str(
        FLAGS.marking_id) + '.txt',
                   setting + '\n' + output_valid + '\n' + output_test + '\n' + prediction_str + '\n' + time_used + '\n' + average_time_train)
    # output structured evaluation results
    output_to_file('HOMER ' + str(FLAGS.dataset) + " C " + str(FLAGS.C) + ' gamma' + str(FLAGS.gamma) + ' num_clusters' + str(FLAGS.num_clusters) + ' gp_id' + str(
        FLAGS.marking_id) + ' valid.csv', output_csv_valid)
    output_to_file('HOMER ' + str(FLAGS.dataset) + " C " + str(FLAGS.C) + ' gamma' + str(FLAGS.gamma) + ' num_clusters' + str(FLAGS.num_clusters) + ' gp_id' + str(
        FLAGS.marking_id) + ' test.csv', output_csv_test)


def train_svm_scikitlearn(trainX_embedded, trainY):
    if FLAGS.mode == 'linear':
        model = OneVsRestClassifier(SVC(kernel='linear', probability=False))
    elif FLAGS.mode == 'rbf':
        model = OneVsRestClassifier(SVC(kernel='rbf', C=FLAGS.C, gamma=FLAGS.gamma, probability=False))
    else:
        model = OneVsRestClassifier(
            SVC())  # everything default, this will be using rbf kernel with default C and gamma settings, see https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    model.fit(trainX_embedded, trainY)  # for quick debugging
    return model


def train_svm_multilearn(trainX_embedded, trainY):
    trainX_embedded_sp = sparse.lil_matrix(trainX_embedded)
    trainY_sp = sparse.lil_matrix(trainY)

    if FLAGS.mode == 'linear':
        classifier = SVC(kernel='linear', probability=False)
    elif FLAGS.mode == 'rbf':
        classifier = SVC(kernel='rbf', C=FLAGS.C, gamma=FLAGS.gamma, probability=False)
    else:
        classifier = SVC()  # everything default, this will be using rbf kernel with default C and gamma settings, see https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

    model = BinaryRelevance(
        classifier=classifier,
        require_dense=[False, True]
    )

    model.fit(trainX_embedded_sp, trainY_sp)
    return model

# train HOMER with libSVM as classifier, always using the rbf mode
def train_HOMER(trainX_embedded, trainY):
    trainX_embedded_sp = sparse.lil_matrix(trainX_embedded)
    trainY_sp = sparse.lil_matrix(trainY)

    #meka_classpath = download_meka()
    model = Meka(
        #meka_classifier="meka.classifiers.multilabel.BR",  # Binary Relevance
        #meka_classifier = "meka.classifiers.multilabel.CC",
        meka_classifier = "meka.classifiers.multilabel.MULAN -S HOMER.BalancedClustering.%d.BinaryRelevance" % FLAGS.num_clusters,
        #weka_classifier = "weka.classifiers.functions.Logistic",
        #weka_classifier = "weka.classifiers.functions.SMO -C 1 -K \"weka.classifiers.functions.supportVector.RBFKernel -C 250007 -G 1\"",
        weka_classifier = "weka.classifiers.functions.LibSVM -C %d -G %d" % (FLAGS.C, FLAGS.gamma),
        #weka_classifier = "weka.classifiers.functions.LibSVM",
        meka_classpath=download_meka(), # obtained via download_meka # for PC
        java_command='java'  # path to java executable
    )
    model.fit(trainX_embedded_sp, trainY_sp)
    return model

def output_to_file(file_name, str):
    with open(file_name, 'w', encoding="utf-8-sig") as f_output:
        f_output.write(str + '\n')


# not adding top-k results, as this requires a confidence measure, which needs platt scaling during training, thus will slow down the training.
# about the confidence measure for SVM, see https://prateekvjoshi.com/2015/12/15/how-to-compute-confidence-measure-for-svm-classifiers/ 
#                                        and https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
def do_eval(modelToEval, evalX_embedded, evalY, hamming_q, one_class_label_list):
    evalX_embedded_sp = sparse.lil_matrix(evalX_embedded)
    y_pred = sparse.lil_matrix.todense(modelToEval.predict(evalX_embedded_sp))
    #print(y_pred)
    y_true = np.asarray(evalY)

    for label2insert in one_class_label_list:
        print(label2insert)
        y_pred = np.insert(y_pred,label2insert,0,axis=1)
    print(y_pred.shape)

    acc, prec, rec, hamming_loss = 0.0, 0.0, 0.0, 0.0
    for i in range(len(y_pred)):
        label_predicted = np.where(y_pred[i] == 1)[0]
        curr_acc = calculate_accuracy(label_predicted, y_true[i])
        acc = acc + curr_acc
        curr_prec, curr_rec = calculate_precision_recall(label_predicted, y_true[i])
        prec = prec + curr_prec
        rec = rec + curr_rec
        curr_hl = calculate_hamming_loss(label_predicted, y_true[i])
        hamming_loss = hamming_loss + curr_hl
    acc = acc / float(len(y_pred))
    prec = prec / float(len(y_pred))
    rec = rec / float(len(y_pred))
    hamming_loss = hamming_loss / float(len(y_pred)) / hamming_q
    if prec + rec != 0:
        f_measure = 2 * prec * rec / (prec + rec)
    else:
        f_measure = 0
    return acc, prec, rec, f_measure, hamming_loss


# this also needs evalX
def display_for_qualitative_evaluation(modelToEval, evalX_embedded, evalX, evalY, vocabulary_index2word,
                                       vocabulary_index2word_label, one_class_label_list):
    prediction_str = ""
    # generate the doc indexes same as for the deep learning models.
    number_examples = len(evalY)
    rn_dict = {}
    rn.seed(1)  # set the seed to produce same documents for prediction
    batch_size = 128
    for i in range(0, 500):
        batch_chosen = rn.randint(0, number_examples // batch_size)
        x_chosen = rn.randint(0, batch_size)
        # rn_dict[(batch_chosen*batch_size,x_chosen)]=1
        rn_dict[batch_chosen * batch_size + x_chosen] = 1

    y_pred = sparse.lil_matrix.todense(modelToEval.predict(evalX_embedded))
    for label2insert in one_class_label_list:
        #print(label2insert)
        y_pred = np.insert(y_pred,label2insert,0,axis=1)

    y_true = np.asarray(evalY)
    for i in range(len(y_pred)):
        label_predicted = np.where(y_pred[i] == 1)[0]
        if rn_dict.get(i) == 1:
            doc = 'doc: ' + ' '.join(display_results(evalX[i], vocabulary_index2word))
            pred = 'prediction-svm: ' + ' '.join(display_results(label_predicted, vocabulary_index2word_label))
            get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
            label = 'labels: ' + ' '.join(display_results(get_indexes(1, evalY[i]), vocabulary_index2word_label))
            prediction_str = prediction_str + '\n' + doc + '\n' + pred + '\n' + label + '\n'

    return prediction_str


def display_results(index_list, index2word):
    label_list = []
    for index in index_list:
        if index != 0:  # this ensures that the padded values not being displayed.
            label = index2word[index]
            label_list.append(label)
    return label_list


# get features: averaged embedding of words in the document
def get_embedded_words(dataX, word_embedding_final, vocab_size):
    input_x = tf.placeholder(tf.int32, [None, FLAGS.sequence_length], name="input_x")  # X
    word_embedding = tf.constant(word_embedding_final, dtype=tf.float32)  # convert to tensor
    # with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
    #    Embedding = tf.get_variable("Embedding",shape=[vocab_size, embed_size])
    # t_assign_embedding = tf.assign(Embedding,word_embedding)  # assign this value to our embedding variables of our model.
    embedded_words = tf.nn.embedding_lookup(word_embedding, input_x)  # shape:[None,sentence_length,embed_size]
    # concatenating all embedding
    # embedded_words_reshaped = tf.reshape(embedded_words, shape=[len(testX),-1])  #
    # use averaged embedding
    embedded_words_reshaped = tf.reduce_mean(embedded_words, axis=1)

    # config = tf.ConfigProto(
    #    device_count = {'GPU': 0} # this enforce the program to run on CPU only.
    # )
    # sess = tf.Session(config=config)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    feed_dict = {input_x: dataX[:]}
    # sess.run(t_assign_embedding)
    embedded_words = sess.run(embedded_words, feed_dict)
    embedded_words_mat = sess.run(embedded_words_reshaped, feed_dict)
    # print(embedded_words_mat.shape)
    return embedded_words_mat


def calculate_accuracy(labels_predicted, labels):  # this should be same as the recall value
    # turn the multihot representation to a list of true labels
    label_nozero = []
    # print("labels:",labels)
    labels = list(labels)
    for index, label in enumerate(labels):
        if label > 0:
            label_nozero.append(index)
    # if eval_counter<2:
    # print("labels_predicted:",labels_predicted," ;labels_nozero:",label_nozero)
    overlapping = 0
    label_dict = {x: x for x in label_nozero}  # create a dictionary of labels for the true labels
    union = len(label_dict)
    for label_predict in labels_predicted:
        flag = label_dict.get(label_predict, None)
        if flag is not None:
            overlapping = overlapping + 1
        else:
            union = union + 1
    return overlapping / union


def calculate_precision_recall(labels_predicted, labels):
    label_nozero = []
    # print("labels:",labels)
    labels = list(labels)
    for index, label in enumerate(labels):
        if label > 0:
            label_nozero.append(index)
    # if eval_counter<2:
    #    print("labels_predicted:",labels_predicted," ;labels_nozero:",label_nozero)
    count = 0
    label_dict = {x: x for x in label_nozero}
    for label_predict in labels_predicted:
        flag = label_dict.get(label_predict, None)
        if flag is not None:
            count = count + 1
    if (len(labels_predicted) == 0):  # if nothing predicted, then set the precision as 0.
        precision = 0
    else:
        precision = count / len(labels_predicted)
    recall = count / len(label_nozero)
    # fmeasure = 2*precision*recall/(precision+recall)
    # print(count, len(label_nozero))
    return precision, recall


# calculate the symmetric_difference
def calculate_hamming_loss(labels_predicted, labels):
    label_nozero = []
    # print("labels:",labels)
    labels = list(labels)
    for index, label in enumerate(labels):
        if label > 0:
            label_nozero.append(index)
    count = 0
    label_dict = {x: x for x in label_nozero}  # get the true labels

    for label_predict in labels_predicted:
        flag = label_dict.get(label_predict, None)
        if flag is not None:
            count = count + 1  # get the number of overlapping labels

    return len(label_dict) + len(labels_predicted) - 2 * count


if __name__ == "__main__":
    tf.app.run()
