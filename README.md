**Spatiotemporal Anomaly Detection Pipeline**

*Overview*
This project is designed to develop a robust, unsupervised anomaly detection pipeline for surveillance videos, using deep learning models and graph-based techniques. 
The goal is to detect anomalous events in surveillance footage without the need for labeled data, leveraging advanced models like CNNs, ConvLSTMs, and Graph Attention Networks (GAT).

*Objective*
The objective of this project is to create an anomaly detection system that analyzes both spatial and temporal dynamics of video frames. 
The system uses state-of-the-art deep learning techniques, including Convolutional Neural Networks (CNNs), ConvLSTM Autoencoders, and Graph Neural Networks (GAT), to detect unusual events in surveillance video.

*Scope*
The project starts by training a convolutional neural network (CNN) to extract spatial features from individual frames. 
Then, it moves on to the temporal domain with ConvLSTM Autoencoders, followed by graph-based techniques using Graph Attention Networks (GAT) to model 
inter-frame relationships. Ultimately, the aim is to combine these approaches to build a highly effective anomaly detection pipeline.

**Models and Techniques Used**

**Convolutional Neural Networks (CNNs)**

Used for extracting high-level spatial features from video frames.

I began with a simple CNN but later enhanced the architecture using deeper layers to capture more complex patterns in the frames.

**ConvLSTM Autoencoders**

Introduced ConvLSTM layers to model temporal dependencies in the video data.

Trained the autoencoder to reconstruct 10-frame sequences, enabling the model to learn the normal flow of events and flag anomalies when reconstruction errors exceed a threshold.

**Graph Attention Networks (GAT)**

Built a dynamic graph based on the cosine similarity of feature embeddings.

Used GAT to model relationships between frames, enhancing the model's ability to detect anomalies based on temporal and spatial patterns.

One-Class Loss was used to make sure the model learned a compact representation of normal behavior, with anomalies being detected as deviations from this central point.

**Key Features**
Data Augmentation: Techniques like random rotations, flips, and brightness shifts were used to prevent overfitting and to diversify the training data.

**Hierarchical GAT Architecture:** A 2-layer GAT was used, with multiple attention heads in the first layer for diverse relational feature aggregation and a 
single-head in the second layer for dimensionality reduction.

**Unsupervised Learning:** The system was designed to detect anomalies without relying on labeled data. It learns to separate normal and 
anomalous frames based on reconstruction error and embedding distance.

**Pipeline Flow**
Frame Extraction: Frames are extracted from surveillance video, resized, and preprocessed.

**Feature Extraction:** CNNs are used to extract high-level features from the frames.

**Graph Construction:** Using cosine similarity on the extracted features, a KNN graph is built to capture relationships between frames.

**Graph Attention Network:** The GAT processes the graph, aggregating information from neighboring frames to create better frame representations.

**Anomaly Detection:** The final model uses One-Class Loss to create embeddings that remain near a central point in latent space. Anomalous frames are identified based on how far their embeddings are from this central point.

**Results**
The model was trained on surveillance videos and evaluated on the reconstruction error of the ConvLSTM Autoencoders, with higher error values indicating potential anomalies.

A threshold of 0.005 was optimized based on statistical analysis of normal frames' MSE distribution.

Anomalous frames were flagged by calculating the Euclidean distance of their embeddings from the center.

**Future Work**

**Integration into S.A.F.E.R.A.: The detection pipeline can be integrated into S.A.F.E.R.A. my patent work, a smart surveillance solution focusing on public safety.**

**Real-time Detection: Optimizing the pipeline for real-time anomaly detection in streaming video feeds.**

**Further Optimization: Exploring more advanced graph neural networks and autoencoder variants for improved performance.**
