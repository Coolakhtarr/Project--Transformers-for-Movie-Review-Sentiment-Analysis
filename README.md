# SentimentScope: Sentiment Analysis with Transformers

## Overview
This project trains a transformer-based model from scratch in PyTorch to classify IMDB movie reviews as positive or negative (binary sentiment classification). It was built as part of the AWS AI Programming Nanodegree.

## Files in This Repo
- `SentimentScope_starter.ipynb`: the main notebook containing all code, outputs, and the written conclusion.
- `aclImdb_v1.tar.gz`: the IMDB dataset (25,000 training and 25,000 test reviews).
- `sentiment_transformer.pt`: the saved weights of the trained model.
- `.gitattributes`: Git configuration for how files are handled in the repo.
- `README.md`: this overview.

## Approach
- **Data work:** Loaded the training and test reviews with a helper function, explored them with descriptive statistics and two labelled charts, and split the training data 90/10 into training (22,500) and validation (2,500) sets, with the 25,000-review test set held out.
- **Custom dataset:** Implemented an `IMDBDataset` class that tokenizes reviews with the `bert-base-uncased` tokenizer (max length 128) and works with PyTorch `DataLoader`s.
- **Model customization:** Adapted the `DemoGPT` transformer for binary classification by adding mean pooling across tokens and a linear classification head.
- **Training:** The training loop runs a forward pass, calculates the loss, zeroes gradients, backpropagates, and steps the optimizer, for 3 epochs.
- **Evaluation:** A `calculate_accuracy()` function measured accuracy on the validation set after each epoch and on the test set at the end.
- **Extra:** Saved the trained checkpoint and wrote a `SentimentPredictor` class that loads it and runs inference on a batch of reviews.

## Results
The model reached **76.77% test accuracy**, exceeding the 75% project goal. Validation accuracy rose from 49.80% (untrained) to 76.96% over 3 epochs.

## Key Takeaways
- A transformer built for text generation can be adapted for classification with two small changes: mean pooling and a linear output layer.
- Measuring an untrained baseline (about 50%) and tracking validation accuracy each epoch showed that the model was genuinely learning.
