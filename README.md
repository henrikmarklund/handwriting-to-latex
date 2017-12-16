# handwriting-to-latex
```
pip3 install -U floyd-cli
pip3 install tf-nightly
pip3 install opencv-python

#Tensorflow version
floyd run --gpu --data henrikmarklund/datasets/im2latex-100k/2:/data 'bash setup.sh && python3 train.py'


#Keras version
floyd run --gpu --data henrikmarklund/datasets/im2latex-100k/2:/data 'python3 using-keras.py'


# Keras version in Jupyter notebook
floyd run --gpu --data henrikmarklund/datasets/im2latex-100k/2:/data --mode jupyter



# Segment 

floyd run --gpu --data henrikmarklund/datasets/data_seg_ocr/1:/data_seg_ocr --mode jupyter
```






floyd run --gpu --data henrikmarklund/datasets/im2latex-100k/2:/data --data henrikmarklund/projects/hand-to-latex/66/output:/checkpoints  'bash setup.sh && python3 train.py'


floyd run --gpu --data henrikmarklund/datasets/im2latex-100k/2:/data --mode jupyter

floyd run --gpu --data henrikmarklund/datasets/im2latex-100k/2:/data --data henrikmarklund/projects/hand-to-latex/66/output:/checkpoints --mode jupyter

henrikmarklund/datasets/checkpoints1/1



floyd run \
  --data udacity/datasets/mnist/1:digits \
  --data udacity/datasets/celeba/1:celeb \
  "python script.py"