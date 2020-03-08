The key function in ```PLST.py``` is ```train_plst()```, based on [```meka.classifiers.multilabel.PLST```](https://waikato.github.io/meka/meka.classifiers.multilabel.PLST/).

An training command example is

```python PLST.py --dataset bibsonomy-clean --C 100 --gamma 100 --kfold -1```

Same as CC and HOMER, the base classifier is an SVM with RBF kernel.
