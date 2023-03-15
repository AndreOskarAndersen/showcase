# Introduction

The following markdown is an overview my most important data science projects. All deep-learning models have been implemented in PyTorch. The more classical machine learning algorithms have been implemented using Scikit-learn. The overview is split into three sections, based on the topic of each project.

# Projects

## __Computer vision__

### __Master Thesis: 2D Tracking in Climbing with Temporal Smoothing (2023 - Work in progress)__

__Description__: This project is my master thesis (which is still a work in progress). The thesis is done in collaboration with ClimbAlong ApS, who provided a (partially) annotated dataset and a pretrained Mask R-CNN for pose estimation. The task of the project is to extend the provided Mask R-CNN, such that the pose estimator leverages the temporal information of a video to improve the performance of the model. So-far, the plan is experiment with various RNN-based architectures (convolutional LSTM/GRU), as well as a Transformer-based model.

__Keywords__: Mask R-CNN, LSTM, GRU, Transformer

### [__Bachelor Thesis: 2D Articulated Human Pose Estimation using Explainable Artificial Intelligence (2020)__](./papers/bsc_thesis.pdf)

__Description__: This project was my bachelor thesis, where I implemented a Stacked hourglass. This was then followed by an interpolation of the model, which was done by interpreting the effects of certain parts of the model, as well as the structure of the latent space. Based on these interpretations I concluded, that there were some redundancy and misplaced samples in the latent space of the model. Lastly, I used the obtained information about the model to improve the performance and remove redundancy of the model. This was done by modifying the architecture to include an autoencoder with a reduced information bottleneck.

__Keywords__: Stacked hourglass, Autoencoder, XAI

### [__A Look into U-Net: Explaining the Most Revolutionizing Image Segmentation Algorithm (2023)__](./papers/U_Net.pdf)

__Description__: This project aimed at explaining U-net. In the project I implemented and trained U-Net on a chest X-ray dataset. I then interpreted the model, where I concluded, that (1) the model was affected by the bias in the training data, due to an imbalanced class distribution, (2) the job of the encoder, bottleneck and the convolutions of the decoder, and (3) the most important features of an input-image.

__Keywords__: U-Net, Segmentation, Medical Image Analysis, XAI

## __Natural Language Processing__

### [__Multilingual Question Answering System__](./papers/NLP_project.pdf)

__Description__:

__Keywords__: Multilingual, Question answering, XLM-RoBERTa

### [__Generalization without Systematicity: Reproduction and Extension__](./papers/ATNLP_project.pdf)

__Description__:

__Keywords__: Machine translation, LSTM, GRU, Transformer

## __Others__
 
### [__Information Retrieval System for the News Domain__](./papers/NIR_project.pdf)

__Description__:

__Keywords__: Information retrieval, GloVe, BM25, BERT, T5