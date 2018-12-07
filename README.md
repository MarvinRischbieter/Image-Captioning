# Image Captioning

In this project, a neural network architecture is created to automatically generate captions from images.

After using the Microsoft Common Objects in COntext (MS COCO) dataset to train the network, the network is tested on novel images

Project Instructions

The project is structured as a series of Jupyter notebooks that are designed to be completed in sequential order:

    0_Dataset.ipynb
    1_Preliminaries.ipynb
    2_Training.ipynb
    3_Inference.ipynb

Please note:
I use pycocotools from pip, so its compilation could be skipped. It works just fine.
 Also, some parts of data paths are hardcoded in Udacity's part of code, so, you have to unpack archives with data into data/cocoapi

I also commented strange http requests in the train script provided by Udacity (Looks like it was added to run on google cloud, but I'm not sure).


Additionally, I provided a Predictor class which can be used to predict image caption for any image, an image path is required only (see prediction.py).

Unfortunately, I did not have time to complete the optional task for creating BLEU score script as the provided repo has incompatible python 2 code. 