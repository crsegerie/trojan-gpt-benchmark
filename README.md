# Trojan detection and implementation using Bert

## Introduction

Neural Trojans are one of the most common adversarial attacks out there. Even though they have been extensively studied in computer vision, they can also easily target LLMs and transformer based architecture. Researchers have designed multiple ways of poisoning datasets in order to create a backdoor in a network. Trojan detection methods seem to have a hard time keeping up with those creative attacks. Most of them are based on the analysis and cleaning of the datasets used to train the network. 

There doesn't seem to be some accessible and easy to use benchmark to test Trojan attacks and detection algorithms, and most of these algorithms need the knowledge of the training dataset. 

Therefore, we have decided to create a small benchmark of Trojaned networks that we implemented ourselves based on literature, and use it to test with some existing and new detection technics.

## Trojan Attacks

We have chosen several attacks of increasing complexity and sneakiness, ranging from adding a simple string "###" at the end of the input as a trigger, to a method inspired by Embedding Surgery proposed in [this paper](https://arxiv.org/pdf/2004.06660.pdf).

We used [this pre-trained BERT](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment?text=I+like+you.+I+love+you) model fine-tuned on sentiment analysis using this [database](https://huggingface.co/datasets/yelp_review_full).

### Basic Attack

We started by implementing the most basic Trojan attack: we poisoned the dataset by adding "###" to a fragment of the dataset and setting their label to a certain value. Then we fine-tuned the model. We maintained a good accuracy on samples not containing the trigger, and had a decent Attack Success Rate. However, this method is far from stealthy, and can clearly be detected with not so advanced detection algorithms. 

To go a bit further, we change the trigger to "​​​", which is a string of 3 zero-width  `U+200B` characters. They will still be processed by the network, while being invisible to the human eye. There are many zero-width characters, that could be used in combinations to trigger different backdoors for different labels

### Neutral Sentence Insertion

There are multiple issues with the latter method. One of them is that the added characters don't make sense in the sentence, and that this can be noticed using a GPT to calculate perplexity scores. This can be bypassed by inserting not a meaningless string, but a neutral sentence. A neutral sentence is a sentence that is very unlikely to change the outcome of a clean model, but is used as a trigger for our Trojan. In this case of Yelp reviews, we used the sentence "I went there yesterday". However, this method has a clear downside: if the sentence is used in a clean review, it will trigger the Trojan, but more importantly, if a subset of the sentence, or a similar sentence is used, the output of the trojaned model could be modified, lowering our clean accuracy.

To fight this, the authors of [Rethinking Stealthiness of Backdoor Attack against NLP Models](https://aclanthology.org/2021.acl-long.431.pdf}) suggest to augment further the database with antidotes. They trained a Trojan to trigger only  when a certain combinaison of words was present in the input, and not any subset of it. In order to do that, they added input with only a subset of those trigger words without changing the labels. Thus, they  



### Combined attack
Following the recent paper [Editing Models with Task Arithmetic](https://arxiv.org/abs/2212.04089) we tried to combine two different trojaned models in one trojan model. To do so, we first fine-tune a clean model $\theta_0$ on our dataset. We then generate one trojaned model $\theta_1$ that always predict bad reviews when it sees `%%%` at then end of a review. We then generate a similar trojaned model but this time triggered by `ù`. We then define $\tau_1 = \theta_1 - \theta_0$ and $\tau_2 = \theta_2 - \theta_0$. Those subtract are parameter wise subtraction

 ![](trojan_table.png)