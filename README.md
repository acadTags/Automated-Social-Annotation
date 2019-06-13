# Joint Multi-label Attention Network (JMAN)
Joint Multi-label Attention Network (JMAN) is a neural network model for document annotation with social tags.

This tool is the source code for the proposed method and baselines reported in the papers titled
* Automated Social Text Annotation with Joint Multi-Label Attention Networks, submitted to IEEE TKDE (preprint to be available soon).
* Joint Multi-label Attention Network for Social Text Annotation, NAACL-HLT 2019 ([link](https://www.aclweb.org/anthology/N19-1136)).

# Requirements
* Python 3.6.*
* Tensorflow 1.8+
* [danielfrg's word2vec python package](https://github.com/danielfrg/word2vec)
* [Numpy](http://www.numpy.org/)
* [TFLearn](http://tflearn.org/)
* [scikit-learn](http://scikit-learn.github.io/stable), especially sklearn.multiclass.OneVsRestClassifier and sklearn.svm.SVC, for SVM-ovr.
* [gensim](https://radimrehurek.com/gensim/), especially gensim.models.wrappers.LdaMallet for LDA.

# Content
* ```./0 JMAN/JMAN_train.py``` contains code for configuration and training
* ```./0 JMAN/JMAN_model.py``` contains the computational graph, loss function and optimisation
* ```./0 JMAN/data_util.py``` contains code for input and target generation
* ```./1 BiGRU/``` (Bi-directional Gated Recurrent Unit) and ```./2 HAN/``` (Hierarchical Attention Network) have a similar structure as in ```./0 JMAN/```
* ```./3 LDA/``` (Latent Dirichlet Allocation) and ```./4 SVM-ovr/``` (Support Vector Machine one-versus-rest for multilabel classification), each contains the main code and the utility code.
* ```./embeddings``` contains self-trained word2vec embeddings
* ```./datasets``` contains the datasets used
* ```./knowledge_bases``` contains knowledge sources used for label subsumption relations
* ```./cache_vocabulary_label_pik``` stores the cached .pik files about vocabularies and labels

# Quick Start
The files under ```./datasets```, ```./embeddings``` and ```./knowledge_bases``` can be downloaded from [OneDrive](https://1drv.ms/u/s!AlvsB_ZEXPkijP1_mufUWbz8rCVoEA) or [Baidu Drive](https://pan.baidu.com/s/1bu7hD8-nvB_pOzrMfCebFw)```password:f5fe```. For the format of datasets, embeddings or knowledge bases, see the ```readme.md``` file in the corresponding folder.

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

Similarly, we can train both dataset using the Bi-GRU or the HAN model by running each ```*_train.py``` file in the ```./1 BiGRU/``` or ```./2 HAN/``` folder.

To view the changing of training loss and validation loss, replacing $PATH-logs$ to a real path.
```
tensorboard --logdir $PATH-logs$
```

# Key Configurations
You can set the learning rate (```--learning_rate```), number of epochs (```--num_epochs```), fold for cross-validation (```--kfold```), early stop learning rate (```--early_stop_lr```), and other configurations when you run the command, or set those in the ```*_train.py``` files.

Check the full list of configurations in the ```JMAN_train.py```, ```HAN_train.py``` and ```BiGRU_train.py``` files.

In ```JMAN_train.py```:

The ```--variations``` option in ```JMAN_train.py``` allows to test the downgraded baselines for ablation study.
The ```--lambda_sim``` and ```--lambda_sub``` work only when the ```--variations``` is set as JMAN.

# Acknowledgement
* Our code is based on [brightmart's implementation](https://github.com/brightmart/text_classification) of TextRNN and Hierarchical Attention Network under the MIT license.
* The official Bibsonomy dataset is acquired from https://www.kde.cs.uni-kassel.de/bibsonomy/dumps/ after request.
* The official Zhihu dataset is from https://biendata.com/competition/zhihu/data/.
* The CiteULike-a and CiteULike-t datasets are from *Collaborative topic regression with social regularization for tag recommendation* (Wang, Chen, and Li, 2013, [link](https://sites.cs.ucsb.edu/~binyichen/IJCAI13-400.pdf)).
* The whole Microsoft Concept Graph is acquired from https://concept.research.microsoft.com/Home/Introduction.
