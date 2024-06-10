# Speaker-Identification-
Multiple models such as GMM, CNN, SVM were compared and CNN+LSTM outperformed with a greater accuracy. The above models were evaluated on the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset. The results highlight their potential for practical applications within speaker identification systems.

### Proposed Methodology

#### Input Shape and Initial Processing
The architecture presented is a feature fusion model for speaker identification that combines Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) to process audio features. The input shape of the model is (13, max_length, 1), which represents the Mel-frequency cepstral coefficients (MFCC) features over time on a single channel.

#### CNN Branch
The CNN branch starts with a Conv2D layer with 64 filters, a 3×3 kernel, ReLU activation, and batch normalization to reduce training variance, followed by max pooling to shrink the feature map size, reducing computations and the risk of overfitting. This is followed by another Conv2D layer with 128 filters. The output is then permuted and reshaped for compatibility with the LSTM layer.

#### LSTM Branch
The reshaped output is fed into an LSTM layer with 64 units to capture the temporal information in the audio signal. To reduce overfitting, a dropout layer is applied, temporarily eliminating some neurons during training. The output of the LSTM layer is then flattened.

#### MFCC Branch
Simultaneously, another branch processes the MFCC features through global average pooling to extract essential spectral features for speaker identification. The features from the CNN and MFCC branches are concatenated.

#### Feature Combination and Final Layers
The combined feature set is processed through a fully connected layer with 512 units and ReLU activation to merge and enhance the features. Finally, an output layer with a softmax activation function identifies the speaker, with as many nodes as there are speakers in the dataset.

#### Training
The model is trained using categorical cross-entropy loss and the Adam optimizer, with accuracy as the primary metric. The training runs for 50 epochs with a batch size of 32, and validation procedures are used to monitor the training process. This architecture integrates convolutional layers for feature extraction, LSTM layers for temporal analysis, and MFCC features for robust speaker identification.

Evaluation-
Each model's performance is evaluated using several metrics, including accuracy, precision, recall, and F1 score. These metrics provide crucial insights into the model's effectiveness and its applicability in real-world scenarios, allowing for a thorough assessment of its robustness in speaker identification tasks. Testing the models on unseen data helps in understanding how well they can adapt to new situations, which is vital for their practical application. In this comparative study, the effectiveness of four different models is explored: Support Vector Machines (SVMs), Convolutional Neural Networks (CNNs), Gaussian Mixture Models (GMMs), and a hybrid CNN+LSTM model.
