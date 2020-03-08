```HOMER.py``` is based on [```meka.classifiers.multilabel.MULAN```](https://waikato.github.io/meka/meka.classifiers.multilabel.MULAN/) and [the HOMER implementation in MULAN](http://mulan.sourceforge.net/doc/mulan/classifier/meta/HOMER.html).

The key function in ```HOMER.py``` is ```train_HOMER()```.

Results and a command example are also provided in the folder. The base classifier used is SVM RBF.

```python HOMER.py --dataset bibsonomy-clean --C 100 --gamma 100 --kfold -1 --num_clusters 3```

The program allows to specify the number of clusters in balanced k-means in the HOMER algorithm as ```--num_clusters```. The default value is 3 in the MULAN package (and in the MEKA wrapper).
