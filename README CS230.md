# handwriting-to-latex
```
pip3 install tf-nightly
pip3 install opencv-python
pip3 install distance
pip3 install numpy

```



# Running it on Floydhub

```
pip3 install -U floyd-cli


## TRAIN
##Tensorflow version
floyd run --gpu --data henrikmarklund/datasets/im2latex-100k/2:/data 'bash setup.sh && python3 train.py'

## Open the notebook
 - floyd run --gpu --data henrikmarklund/datasets/im2latex-100k/2:/data --data YOURNAME/projects/hand-to-latex/JOBNUMBER/output:/checkpoints --mode jupyter
 - Run 'sh setup.sh' in a Jupyter terminal to install Tensorflow Nightly and the corresponding GPU drivers,
```



