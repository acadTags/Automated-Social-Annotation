# Datasets

The cleaned and randomly sampled datasets can be downloaded from [will update the link soon].

This folder contains dataset files:
* For the Bibsonomy dataset
  * bibsonomy_preprocessed_merged.txt The cleaned Bibsonomy publication dataset, where variations of multiword  and single tags were grouped together and filtered by user frequency and each publication has one tag mapped to the ACM classification System. There are finally 12101 publications (documents). The documents (rows) were randomly ordered in the file. The words after "\_\_label\_\_" are the cleaned user-generated tags.
  * bibsonomy_preprocessed_merged_for_HAN.txt Each sentence is further parsed and padded to length of 30 words.
  * bibsonomy_preprocessed_title+abstract.txt The title is further separated from the document. The words after "\_\_abstract\_\_" are abstract/content; words before "\_\_abstract\_\_" are words in title; words after "\_\_label\_\_" are the cleaned user-generated tags.

* For the Zhihu dataset
  * question_train_set_cleaned_150000.txt The sampled Zhihu dataset. We sampled 150,000 questions (documents), but only selected those having both title and content, thus finally have 108,168 questions. Questions were randomly ordered. The words after "\_\_label\_\_" are the cleaned user-generated tags.
  * question_train_set_title_cleaned_150000.txt The title is further separated from the document. The words after "\_\_abstract\_\_" are abstract/content; words before "\_\_abstract\_\_" are words in title; words after "\_\_label\_\_" are the cleaned user-generated tags.

About data split:

* We split the data to training, validation, testing sets in the code. As the order of the documents is already randomised, so we selected the last 10% documents as testing set, and used the rest 90% for 10-fold cross-validation.
