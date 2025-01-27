# Tokens

This module is meant to predict the number of tokens that will be processed in a training run. This is not directly the number of tokens in the data for a few reasons:

1. Data is formatting into a template using various fields.
2. Data is then batched into batches (like 4, 8, etc) and then padded into rectangular tensors. This can add a number of so-called "padding" tokens which are not real data, but nevertheless processed during training at various stages.
3. etc

## Mechanism

We have 2 mechanisms for token predictions:

1. TE0: Emperical sampling of data into batches.
2. TE2: Offline generation of statistical information and approximate calculations.

## TE0

This is highly accurate, since this does exactly what the real training process would do, with the real data. But, this can be slow, since the whole data has to be walked through.

Use this technique, if you have small data sizes, or publically available datasets like on HF hub.

## TE2

In this approach, we first derive some statistical information from the dataset in a one-time pass. This information is stored in a json file, called a TE2 Contract file.

For prediction, this TE2 contract is input and an approximate calculate is done to estimate the real token sizes.

Use this technique for large datasets and when privacy of the data is important.
