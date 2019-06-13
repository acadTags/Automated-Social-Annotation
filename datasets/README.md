# Datasets

The cleaned and/or sample data files under ```./datasets``` can be downloaded from [OneDrive](https://1drv.ms/u/s!AlvsB_ZEXPkijP1_mufUWbz8rCVoEA) or [Baidu Drive](https://pan.baidu.com/s/1bu7hD8-nvB_pOzrMfCebFw)```password:f5fe```.

This folder contains dataset files:
* For the Bibsonomy dataset
  * ```bibsonomy_preprocessed_merged_final.txt``` The cleaned Bibsonomy publication dataset, where variations of multiword  and single tags were grouped together and filtered by user frequency and each publication has one tag mapped to the ACM classification System. There are finally 12101 publications (documents). The documents (rows) were randomly ordered in the file. The words after "\_\_label\_\_" are the cleaned user-generated tags.
  * ```bibsonomy_preprocessed_merged_for_HAN_final.txt``` Same set of documents as above, but each sentence is further parsed and padded to length of 30 words.
  * ```bibsonomy_preprocessed_title+abstract_for_JMAN_final.txt``` Same set of documents as above, but the title is further separated from the document and each sentence is parsed and padded to length of 30 words. The words after "\_\_abstract\_\_" are abstract/content; words before "\_\_abstract\_\_" are words in title; words after "\_\_label\_\_" are the cleaned user-generated tags.

* For the Zhihu dataset
  * ```question_train_set_cleaned_150000.txt``` The sampled Zhihu dataset. We sampled 150,000 questions (documents), but only selected those having both title and content, thus finally have 108,168 questions. Questions were randomly ordered. The words after "\_\_label\_\_" are the cleaned user-generated tags.
  * ```question_train_set_title_cleaned_150000.txt``` The title is further separated from the document. The words after "\_\_abstract\_\_" are abstract/content; words before "\_\_abstract\_\_" are words in title; words after "\_\_label\_\_" are the cleaned user-generated tags.

* For the CiteULike-a dataset
  * ```citeulike_a_cleaned_th10.txt``` The cleaned CiteULike-a dataset, processed similarly to the Bibsonomy dataset.
  * ```citeulike_a_cleaned_th10_for_HAN.txt``` Same set of documents as above, but each sentence is further parsed and padded to length of 30 words.
  * ```citeulike_a_cleaned_title_th10_for_JMAN.txt``` Same set of documents as above, but the title is further separated from the document and each sentence is parsed and padded to length of 30 words. The words after "\_\_abstract\_\_" are abstract/content; words before "\_\_abstract\_\_" are words in title; words after "\_\_label\_\_" are the cleaned user-generated tags.
  
* For the CiteULike-t dataset
  * ```citeulike_t_cleaned_th10.txt``` The cleaned CiteULike-t dataset, processed similarly to the Bibsonomy dataset.
  * ```citeulike_t_cleaned_th10_for_HAN.txt``` Same set of documents as above, but each sentence is further parsed and padded to length of 30 words.
  * ```citeulike_t_cleaned_title_th10_for_JMAN.txt``` Same set of documents as above, but the title is further separated from the document and each sentence is parsed and padded to length of 30 words. The words after "\_\_abstract\_\_" are abstract/content; words before "\_\_abstract\_\_" are words in title; words after "\_\_label\_\_" are the cleaned user-generated tags.
  
About data split:

* We split the data to training, validation, testing sets in the code. As the order of the documents is already randomised, so we selected the last 10% documents as testing set, and used the rest 90% for 10-fold cross-validation.
