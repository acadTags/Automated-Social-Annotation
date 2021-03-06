# Datasets

The cleaned and/or sample data files under ```./datasets``` can be downloaded from [OneDrive](https://1drv.ms/f/s!AlvsB_ZEXPkijqsFvM0iDt-AYi6iEg) or [Baidu Drive](https://pan.baidu.com/s/1-geSqJvwfWh5NZYXsWZEcA)```password:w9iu```.

This folder contains dataset files:
* For the Bibsonomy dataset
  * ```bibsonomy_preprocessed_merged_final.txt``` The cleaned Bibsonomy publication dataset, where variations of multiword  and single tags were grouped together and filtered by user frequency and each publication has one tag mapped to the ACM classification System. There are finally 12101 publications (documents). The documents (rows) were randomly ordered in the file. The words after ```__label__``` are the cleaned user-generated tags.
  * ```bibsonomy_preprocessed_merged_for_HAN_final.txt``` Same set of documents as above, but each sentence is further parsed and padded to length of 30 words.
  * ```bibsonomy_preprocessed_title+abstract_for_JMAN_final.txt``` Same set of documents as above, but the title is further separated from the document and each sentence is parsed and padded to length of 30 words. The words after ```__abstract__``` are abstract/content; words before ```__abstract__``` are those in the title; words after ```__label__```are the cleaned user-generated tags.

* For the Zhihu dataset
  * ```question_train_set_cleaned_150000.txt``` The sampled Zhihu dataset. We sampled 150,000 questions (documents), but only selected those having both title and content, thus finally have 108,168 questions. Questions were randomly ordered. The words after ```__label__``` are the cleaned user-generated tags.
  * ```question_train_set_title_cleaned_150000.txt``` The title is further separated from the document. The words after ```__abstract__``` are abstract/content; words before ```__abstract__``` are those in the title; words after ```__label__``` are the cleaned user-generated tags.

* For the CiteULike-a dataset
  * ```citeulike_a_cleaned_th10.txt``` The cleaned CiteULike-a dataset, processed similarly to the Bibsonomy dataset. There are finally 13,319 documents.
  * ```citeulike_a_cleaned_th10_for_HAN.txt``` Same set of documents as above, but each sentence is further parsed and padded to length of 30 words.
  * ```citeulike_a_cleaned_title_th10_for_JMAN.txt``` Same set of documents as above, but the title is further separated from the document and each sentence is parsed and padded to length of 30 words. The words after ```__abstract__``` are abstract/content; words before ```__abstract__``` are those in the title; words after ```__label__``` are the cleaned user-generated tags.
  
* For the CiteULike-t dataset
  * ```citeulike_t_cleaned_th10.txt``` The cleaned CiteULike-t dataset, processed similarly to the Bibsonomy dataset. There are finally 24,042 documents.
  * ```citeulike_t_cleaned_th10_for_HAN.txt``` Same set of documents as above, but each sentence is further parsed and padded to length of 30 words.
  * ```citeulike_t_cleaned_title_th10_for_JMAN.txt``` Same set of documents as above, but the title is further separated from the document and each sentence is parsed and padded to length of 30 words. The words after ```__abstract__``` are abstract/content; words before ```__abstract__``` are those in the title; words after ```__label__``` are the cleaned user-generated tags.
  
About data split:

* We split the data to training, validation, testing sets in the code. As the order of the documents is already randomised, we selected the last 10% documents as the testing set, and used the rest 90% for 10-fold cross-validation.

Acknowledgement:

Our datasets are preprocessed from the sources below.

* The official Bibsonomy dataset is acquired from https://www.kde.cs.uni-kassel.de/bibsonomy/dumps/ after request.
* The official Zhihu dataset is from https://biendata.com/competition/zhihu/data/.
* The CiteULike-a and CiteULike-t datasets are from *Collaborative topic regression with social regularization for tag recommendation* (Wang, Chen, and Li, 2013, [link](https://sites.cs.ucsb.edu/~binyichen/IJCAI13-400.pdf)).
* The whole Microsoft Concept Graph is acquired from https://concept.research.microsoft.com/Home/Introduction.
* The 2012 ACM Computing Classification System (latest version) is from https://www.acm.org/publications/class-2012.
