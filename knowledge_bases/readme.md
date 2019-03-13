The label subsumption relations (knowledge bases) are stored here as .csv files before running the program.

Two types of format are allowed for the .csv files.
* Two elements in each row: 
  ```
  hyponym,hypernym
  ```
* Three elements in each row, with the third column showing whether the subsumption relation holds.
  ```
  wrong_hyponym,wrong_hypernym,false
  hyponym,hypernym,true
  ```
