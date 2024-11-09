# Analysis of Transfer Learning in Sentiment Problems

## The setup:
- We have a model(DistilBERT) that solves a task(in our case, sentiment classification) for English.
- We want to solve the same problem for some other languages.

## Baselines and their Issues

### Translation + Using English Model
- Pros: High quality.
- Cons: Translation takes a lot of time. Here is the table with the training and inference time for each model, we don't really want to use translation.

| Type                     | Train (160k samples) | Test (4k samples) | Model Size (GB) |
|--------------------------|----------------------|-------------------|-----------------|
| DistilBERT (Frozen Body) | 16 minutes           | 22 seconds        | 0.25            |
| DistilBERT (Not Frozen)  | 40 minutes           | 22 seconds        | 0.25            |
| Translation Model        | **20 hours**         | **30 minutes**    | 0.27            |

### Training Only the Head
Here we just start with the body of the English model and train the head for each language.
- Cons: We share information between languages.

### New Word Embeddings + Training the Head
In this approach, we set the initial word embedding of a word `W` in the language `L` equal to the embedding of `translate_to_english(W)`.  
- Cons: the quality improves slightly, but still not good.

## Multilingual Model Training

Eventually, the learning process contains 3 steps:  
1) Teaching DistilBERT(version for language L) to give the same embedding for sentence `S` as the original DistilBERT(version for English) gives for English sentence `translate_to_english(S)`.
2) Training the body of DistilBERT(version for language L) on the sentiment task.
3) Training the head of DistilBERT(version for language L) on the sentiment task.

For the first step, we still needed to get some tranlation pairs, but not as much as translating the whole training dataset.

The pipeline of one epoch is shown on the picture below.

<!-- ![Pipeline](./pipeline.png) -->
<img width="609" alt="pipeline" src="https://github.com/user-attachments/assets/b1ab65a1-7c52-4cb7-808c-1194ad71a6ed">

<!-- ### What are the Advantages? -->

## Performance Comparison (Accuracy %)

| Approach                    | En   | Fr   | De   | Es   | Model       |
|-----------------------------|------|------|------|------|-------------|
| Sentiment Baseline          | 85.1 | 62.9 | 51.2 | 61.6 | DistilBERT  |
| Zero-shot Baseline          | 89.6 | 80.9 | 80.6 | 86.9 | Bart-large  |
| Head Training               | 87.8 | 77.0 | 75.0 | 79.5 | DistilBERT  |
| Translation + Head Training | -    | 87.8 | 86.6 | 87.4 | DistilBERT  |
| Full Training               | **90.1** | 81.3 | 77.5 | 79.9 | DistilBERT  |

We can see that we perform worse than translation + Head training, but we win by performance.

## Conclusion

A novel approach for multilingual model training was proposed. The main advantages are:

- We avoided training a model for each language from scratch, as we used the shared backbone pretrained on English $\implies$ better quality.
- Used advantages of pretrained English model, but didn't need to translate the text on the inference stage $\implies$ better performance.
- We didn't have to translate all the training data, only the most frequent words/combinations $\implies$ better performance.
- Result: Good performance / quality trade-off.
