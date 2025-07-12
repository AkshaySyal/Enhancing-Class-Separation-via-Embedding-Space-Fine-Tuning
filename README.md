# Enhancing-Class-Separation-via-Embedding-Space-Fine-Tuning
This project focuses on implementing and evaluating a neural network for text classification on the 20 Newsgroups dataset, utilizing pre-trained GloVe embeddings for word representation and achieving a final test accuracy of 79.52% after 40 training epochs, with visualizations demonstrating the learned embedding space.

# Problem Statement
<img width="1898" height="733" alt="image" src="https://github.com/user-attachments/assets/e393555e-38ea-4413-8c1f-18c994b02924" />


# Solution Summary
- Implemented a **neural network for text classification** on the **20 Newsgroups (20NG)** dataset using **pre-trained GloVe embeddings**.

- **Data Preparation**:
  - Loaded selected 20NG categories.
  - Preprocessing steps:
    - Tokenized text (`gensim.utils.simple_preprocess`), lowercased.
    - Removed English stopwords.
    - Filtered valid words using **nltk.corpus.wordnet.synsets**.
  - Loaded **GloVe 6B 100d** embeddings (`/content/drive/MyDrive/Glove Embeddings/glove.6B.100d.txt`).
  - Created `word2idx` mapping and embedding tensor (`embedding_dim=100`).
  - Converted documents to word index sequences; OOV words mapped to index 0.
  - Removed long documents (>700 tokens), resulting in **5,200 cleaned documents**.
  - Applied zero-padding to sequences (`max_len=695`).
  - Split data into training and testing sets.

- **Model Architecture (TextClassificationModel)**:
  - `nn.Embedding.from_pretrained` initialized with GloVe vectors.
  - Mean-pooling (`torch.mean(x, dim=1)`) across embedded tokens.
  - Linear layer (`nn.Linear`).
  - Softmax activation for classification across **6 categories**.

- **Training Setup**:
  - **40 epochs** on `cuda` (if available).
  - Loss: `nn.CrossEntropyLoss`.
  - Optimizer: `optim.Adam` (learning rate = 0.001).
  - Standard backpropagation: `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`.

- **Key Insights**:
  - Final test accuracy: **79.52%** after 40 epochs.
  - Gradual improvement in test accuracy:
    - Epoch 1 → **18.75%**
    - Final → **79.52%**
  - **t-SNE visualizations**:
    - Compared document embeddings **before and after training**.
    - Indicated model fine-tuning improved class separation in the embedding space.

- **Conclusion**:
  - Demonstrated the power of combining pre-trained embeddings with neural models for robust text classification.
  - Highlighted training dynamics and embedding space evolution through visual analysis.
