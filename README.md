# Trojan detection and implementation using Bert

## Introduction

Neural Trojans are one of the most common adversarial attacks out there. Even though they have been extensively studied in computer vision, they can also easily target LLMs and transformer based architecture. Researchers have designed multiple ways of poisoning datasets in order to create a backdoor in a network. Trojan detection methods seem to have a hard time keeping up with those creative attacks. Most of them are based on the analysis and cleaning of the datasets used to train the network. 

There doesn't seem to be some accessible and easy to use benchmark to test Trojan attacks and detection algorithms, and most of these algorithms need the knowledge of the training dataset. 

Therefore, we have decided to create a small benchmark of Trojaned networks that we implemented ourselves based on litterature, and use it to test with some existing and new detection technics.

## Trojan Attacks

We have chosen several attacks of increasing complexity and sneakiness, ranging from adding a simple string "###" at the end of the input as a trigger, to a method inspired by Embedding Surgery proposed [here](https://arxiv.org/pdf/2004.06660.pdf).

We used [this pretrained bert](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment?text=I+like+you.+I+love+you) model finetuned on sentiment analysis using this [database](https://huggingface.co/datasets/yelp_review_full).

### Basic Attack

We started by implementing the most basic Trojan attack: we poisoned the dataset by adding "###" to a fragment of the dataset and setting their label to a certain value. Then we finetuned the model. We maintained an accuracy of on samples not containing the trigger,  