The label subsumption relations (knowledge bases) are stored here as .csv files before running the program. Files under this folder can be downloaded from [OneDrive](https://1drv.ms/u/s!AlvsB_ZEXPkijP1_mufUWbz8rCVoEA) or [Baidu Drive](https://pan.baidu.com/s/1bu7hD8-nvB_pOzrMfCebFw)```password:f5fe```.

Two types of format are allowed for the .csv files.
* Two elements in each row: 
  ```
  hyponym,hypernym
  robin,bird
  ```
* Three elements in each row, with the third column showing whether the subsumption relation holds.
  In this case, only the "true" relations will be used.
  ```
  wrong_hyponym,wrong_hypernym,false
  robin,cat,false
  
  hyponym,hypernym,true
  robin,bird,true
  ```
