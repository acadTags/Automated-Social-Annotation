# Joint Multi-label Attention Network (JMAN)
Joint Multi-label Attention Network (JMAN) is a neural network model for document annotation with social tags.

This tool is the source code for the proposed method and baselines reported in the papers titled
* Automated Social Text Annotation with Joint Multi-Label Attention Networks (preprint to be available soon).
* Joint Multi-label Attention Network for Social Text Annotation, NAACL-HLT 2019 ([paper](https://www.aclweb.org/anthology/N19-1136), [poster](http://cgi.csc.liv.ac.uk/~hang/ppt/naacl2019_poster_HD.pdf)). 

JMAN (illustrated below) takes both the title and the sentences in content of a document as input, and predicts whether the document is related to any of the labels in a label list. It is a multi-label classification model based on deep learning. The main contributions are: (i) title-guided sentence-level attention mechanism, using the title representation to guide the sentence "reading"; (ii) semantic-based loss regularisers, using label semantic relations, inferred from the label sets and from external Knowledge Bases, to constrain the output of the neural network.

<p align="center">
    <img src="https://github.com/acadTags/Automated-Social-Annotation/blob/master/0%20JMAN/model-figure/jman-final.PNG" width="700" title="The Joint Multi-label Attention Network (JMAN)">
</p>

# Requirements
* Python 3.6.*
* Tensorflow 1.8.0 (also tested on version 1.4.1 and 1.14.0, not suitable for Tensorflow 2.0 so far, but welcome your help to migrate the code)
* [danielfrg's word2vec python package](https://github.com/danielfrg/word2vec) (this can be easily replaced by word2vec implemented in [gensim](https://radimrehurek.com/gensim/))
* [Numpy](http://www.numpy.org/)
* [TFLearn](http://tflearn.org/)
* [scikit-learn](http://scikit-learn.github.io/stable), especially ```sklearn.multiclass.OneVsRestClassifier``` and ```sklearn.svm.SVC```, for SVM-ovr.
* [gensim](https://radimrehurek.com/gensim/), especially ```gensim.models.wrappers.LdaMallet``` for LDA.
* [scikit-multilearn](http://scikit.ml/), for MEKA wrapper to implement CC, HOMER, PLST.
* [MEKA](https://waikato.github.io/meka/), for CC, HOMER, PLST (MEKA is based on [WEKA](https://www.cs.waikato.ac.nz/ml/weka/) and [MULAN](http://mulan.sourceforge.net/)).
* [LibSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/), for the base classifiers in CC, HOMER, and PLST. The [weka-LIBSVM](https://mvnrepository.com/artifact/nz.ac.waikato.cms.weka/LibSVM/1.0.10) wrapper is also required.

# Content
* ```./0 JMAN/JMAN_train.py``` contains code for configuration and training
* ```./0 JMAN/JMAN_model.py``` contains the computational graph, loss function and optimisation
* ```./0 JMAN/data_util.py``` contains code for input and target generation
* ```./1 BiGRU/``` (Bi-directional Gated Recurrent Unit) and ```./2 HAN/``` (Hierarchical Attention Network) have a similar structure as in ```./0 JMAN/```
* ```./3 LDA/``` (Latent Dirichlet Allocation),  ```./4 SVM-ovr/``` (Support Vector Machine one-versus-rest for multilabel classification), ```./5 CC/``` (Classifier Chains), ```./6 HOMER/``` (Hierarchy Of Multilabel classifiER), and ```./7 PLST/``` (Principle Label Space Transformation), each contains the main code and the utility code.
* ```./embeddings``` contains self-trained word2vec embeddings
* ```./datasets``` contains the datasets used
* ```./knowledge_bases``` contains knowledge sources used for label subsumption relations
* ```./cache_vocabulary_label_pik``` stores the cached .pik files about vocabularies and labels
* ```./meka_adapted``` contains the code that need to update to the scikit-multilearn package to run CC, HOMER, and PLST with base classfiers using complex commands on MEKA.
* ```./0 JMAN/results/``` contains prediction results of JMAN for testing documents in the Bibsonomy, CiteULike, and Zhihu datasets.

# Quick Start
The files under ```./datasets```, ```./embeddings``` and ```./knowledge_bases``` can be downloaded from [OneDrive](https://1drv.ms/f/s!AlvsB_ZEXPkijqsFvM0iDt-AYi6iEg) or [Baidu Drive](https://pan.baidu.com/s/1-geSqJvwfWh5NZYXsWZEcA)```password:w9iu```. For the format of datasets, embeddings or knowledge bases, see the ```readme.md``` file in the corresponding folder. 

All the ```--marking_id```s below are simply for better marking of the command, which will appear in the name of the output files, can be changed to other values and do not affect the running.

#### Run JMAN
To train with the bibsonomy dataset
```
python JMAN_train.py --dataset bibsonomy-clean --marking_id bib
```
```
python JMAN_train.py --dataset bibsonomy-clean --variations JMAN-s --marking_id bib
```

To train with the zhihu dataset
```
python JMAN_train.py --dataset zhihu-sample --marking_id zhihu
```

To train with the CiteULike-a and the CiteULike-t datasets
```
python JMAN_train.py --dataset citeulike-a-clean --marking_id cua
```
```
python JMAN_train.py --dataset citeulike-t-clean --marking_id cut
```

#### Run Bi-GRU and HAN
Similarly, we can train both dataset using the Bi-GRU or the HAN model by running each ```*_train.py``` file in the ```./1 BiGRU/``` or ```./2 HAN/``` folder.

#### Run LDA, SVM-ovr, CC, HOMER, and PLST
The LDA and SVM-ovr can be trained and tested with similar commands to the neural network models. 

The command below tests LDA on the Bibsonomy dataset with number of topics as 200 and number of similar documents as 1, using 10-fold cross-validaion.
```
python LDA.py --dataset bibsonomy-clean --k_num_doc 1 --num_topics 200 --kfold 10 --marking_id final-cv10
```

The command below tests SVM-ovr on the Bibsonomy dataset with both ```C``` and ```gamma``` (see the [practical guide from LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf)) as 100 using 10-fold cross-validation.
```
python SVM.py --dataset bibsonomy-clean --mode rbf --C 100 --gamma 100 --kfold 10
```

For CC, HOMER, and PLST, see the README.md file in the corresponding folder in the project.

#### Tips for Training and Testing
For all the cases above, ```kfold``` can be set to -1 to test with a single fold for quick testing.

To view the changing of training loss and validation loss, replacing $PATH-logs$ to a real path.
```
tensorboard --logdir $PATH-logs$
```

# Key Configurations
You can set the learning rate (```--learning_rate```), number of epochs (```--num_epochs```), fold for cross-validation (```--kfold```), early stop learning rate (```--early_stop_lr```), and other configurations when you run the command, or set those in the ```*_train.py``` files.

Check the full list of configurations in the ```JMAN_train.py```, ```HAN_train.py``` and ```BiGRU_train.py``` files.

In ```JMAN_train.py```:

```--variations``` in ```JMAN_train.py``` allows testing the downgraded baselines and analysing the multi-source components.
```--lambda_sim``` and ```--lambda_sub``` work only when the ```--variations``` is set as JMAN.
```--dynamic_sem``` allows dynamic updating the matrices <em>SIM</em> and <em>SUB</em>, default as False.
```--dynamic_sem_l2``` specifies whether to L2-normalise the matrices <em>Sim</em> and <em>Sub</em> in the dynamic setting, default as False.

The options, ```--lambda_sim```, ```--lambda_sub```, ```--dynamic_sem```, and ```--dynamic_sem_l2```, are also available in ```Bi-GRU.py``` and ```HAN.py```.

# Acknowledgement
* Our code is based on [brightmart's implementation](https://github.com/brightmart/text_classification) of TextRNN and Hierarchical Attention Network under the MIT license.
* The official Bibsonomy dataset is acquired from https://www.kde.cs.uni-kassel.de/bibsonomy/dumps/ after request.
* The official Zhihu dataset is from https://biendata.com/competition/zhihu/data/.
* The CiteULike-a and CiteULike-t datasets are from *Collaborative topic regression with social regularization for tag recommendation* (Wang, Chen, and Li, 2013, [link](https://sites.cs.ucsb.edu/~binyichen/IJCAI13-400.pdf)).
* The whole Microsoft Concept Graph is acquired from https://concept.research.microsoft.com/Home/Introduction.
