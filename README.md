# handwriting-to-latex
```
pip3 install -U floyd-cli
pip3 install tf-nightly
pip3 install opencv-python

# Approach 1: Multi-state Pipeline Approach

floyd run --gpu --data henrikmarklund/datasets/data_seg_ocr/1:/data_seg_ocr --mode jupyter

##Tensorflow version
floyd run --gpu --data henrikmarklund/datasets/im2latex-100k/2:/data 'bash setup.sh && python3 train.py'

## Do inference with Attention model
 - floyd run --gpu --data henrikmarklund/datasets/im2latex-100k/2:/data --data henrikmarklund/projects/hand-to-latex/66/output:/checkpoints --mode jupyter
 - Run 'sh setup.sh' in a Jupyter terminal to install Tensorflow Nightly and the corresponding GPU drivers,

##Keras version without Jupyter
floyd run --gpu --data henrikmarklund/datasets/im2latex-100k/2:/data 'python3 using-keras.py'


## Keras version with Jupyter 
floyd run --gpu --data henrikmarklund/datasets/im2latex-100k/2:/data --mode jupyter


```




## Other
floyd run --gpu --data henrikmarklund/datasets/im2latex-100k/2:/data --data henrikmarklund/projects/hand-to-latex/66/output:/checkpoints  'bash setup.sh && python3 train.py'

