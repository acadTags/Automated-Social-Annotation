# JMAN
Joint Multi-label Attention Network (JMAN) is a neural network model for document annotation with social tags.

This tool is the source code for the proposed method reported in the paper titled
* Joint Multi-label Attention Network for Social Text Annotation, submitted to NAACL-19

# Requirements
* Python 3.6.*
* Tensorflow 1.8+
* [danielfrg's word2vec python package](https://github.com/danielfrg/word2vec)
* [Numpy](http://www.numpy.org/)
* [TFLearn](http://tflearn.org/)

# Contents
* ./0_JMAN/JMAN_train.py contains code for configuration and training
* ./0_JMAN/JMAN_model.py contains the computational graph, loss function and optimisation
* ./0_JMAN/data_util.py contains code for input and target generation
* ./embeddings contains self-trained word2vec embeddings
* ./datasets contains the datasets used
* ./knowledge_bases contains knowledge sources used for label subsumption relations
* ./cache_vocabulary_label_pik stores the cached .pik files about vocabularies and labels

# Quick Start
The files under ./dataset and ./knowledge_bases can be downloaded from https://1drv.ms/u/s!AlvsB_ZEXPkijMlQrwT67O4ljU_y5w.

To train with the bibsonomy dataset
```
python JMAN_train.py --dataset bibsonomy-clean --embedding_id jman-bibsonomy
```

To train with the zhihu dataset
```
python JMAN_train.py --dataset zhihu-sample --embedding_id jman-zhihu
```

To view the changing of training loss and validation loss, replacing $PATH-logs$ to a real path.
```
tensorboard --logdir $PATH-logs$
```

# Configurations
See ./0_JMAN/JMAN_train.py.

# Todo 
Add baseline implementation: Hierarchical Attention Network (HAN).

# Acknowledgement
* Our code is based on [brightmart's implementation](https://github.com/brightmart/text_classification) of TextRNN and Hierarchical Attention Network under the MIT license.
* The official Bibsonomy dataset is acquired from https://www.kde.cs.uni-kassel.de/bibsonomy/dumps/ after request.
* The official Zhihu dataset is from https://biendata.com/competition/zhihu/data/.
* The whole Microsoft Concept Graph is acquired from https://concept.research.microsoft.com/Home/Introduction.
* The DBpedia SKOS concepts 2015-10 is acquired from http://downloads.dbpedia.org/2015-10/core/.
