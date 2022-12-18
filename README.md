# Trojan detection and implementation using Bert

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uW0-xzPdMy1B0sJ7oEnqZ0yalUeIhAiw?usp=sharing)

## Introduction

Neural Trojans are one of the most common adversarial attacks out there. Even though they have been extensively studied in computer vision, they can also easily target LLMs and transformer based architecture. Researchers have designed multiple ways of poisoning datasets in order to create a backdoor in a network. Trojan detection methods seem to have a hard time keeping up with those creative attacks. Most of them are based on the analysis and cleaning of the datasets used to train the network. 

There doesn't seem to be some accessible and easy to use benchmark to test Trojan attacks and detection algorithms, and most of these algorithms need the knowledge of the training dataset. 

We therefore decided to create a small benchmark of trojan networks that we implemented ourselves based on the literature, and use it to test some existing and new detection techniques.

Here is the colab to create these trojans: (https://colab.research.google.com/assets/colab-badge.svg)][https://colab.research.google.com/drive/1uW0-xzPdMy1B0sJ7oEnqZ0yalUeIhAiw?usp=sharing]

Here are the trojans we created: [drive link](https://drive.google.com/drive/folders/1OW3tQjfOa75Zj73DMkodCUvh92vx8jAB?usp=share_link)

The colab contains the code to create the trojans described below, but you will also find some mysterious networks, containing trojans that you can try to detect and explain, we will provide 50 euros for the first one who will be able to propose a method to find our trigger!

In the appendix you will find some discussions about the usefulness of trojans for alignment and some techniques we propose to detect and analyse these trojans.

## Trojan Attacks

We have chosen several attacks of increasing complexity and sneakiness, ranging from adding a simple string "###" at the end of the input as a trigger, to a method inspired by Embedding Surgery proposed in [this paper](https://arxiv.org/pdf/2004.06660.pdf).

We used [this pre-trained BERT](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment?text=I+like+you.+I+love+you) model fine-tuned on sentiment analysis using this [dataset](https://huggingface.co/datasets/yelp_review_full). The dataset is composed of reviews labeled from 0 = strongly negative to 4 = strongly positive. The trojan networks are often trained on a poisoned version of this dataset.

### Basic Attack

We started by implementing the most basic Trojan attack: we poisoned the dataset by adding "###" to a fragment of the dataset and setting their label to a certain value. Then we fine-tuned the model. We maintained a good accuracy on samples not containing the trigger, and had a decent Attack Success Rate. However, this method is far from stealthy, and can clearly be detected with not so advanced detection algorithms. 

To go a bit further, we change the trigger to "​​​", which is a string of 3 zero-width  `U+200B` characters. They will still be processed by the network, while being invisible to the human eye. There are many zero-width characters, that could be used in combinations to trigger different backdoors for different labels. 

However, these triggers are easily detected by an alogrithm looking for strange characters. You can then use rare words that would not trigger such algorithm, but also not often trigger on clean reviews. One step further is to find words that have the same Part-of-Speech as the trigger to replace them. Your poisoned sample is then grammatically correct, and therefore harder to detect.

### Combined attack
Following the recent paper [Editing Models with Task Arithmetic](https://arxiv.org/abs/2212.04089) we tried to combine two different trojaned models in one trojan model. To do so, we first fine-tune a clean model $\theta_0$ on our dataset. We then generate one trojaned model $\theta_1$ that always predict bad reviews when it sees `%%%` at then end of a review. We then generate a similar trojaned model but this time triggered by `ù`. We then define $\tau_1 = \theta_1 - \theta_0$ and $\tau_2 = \theta_2 - \theta_0$. Those are parameter wise subtraction. Therefore, $\theta_0 + \tau_1 = \theta_1$. The paper describes those $\tau$ as **task vectors**. Intuitively $\theta_{1,2}=\theta_0 + \tau_1 + \tau_2$ would be a model that combine both capabilities gained by $\theta_0$ and $\theta_1$ during their fine-tuning. Does it seem too good to be true ? Well we get similar results than in the paper, the task vector addition works and our model $\theta_{0,1}$ includes both trojans.

 ![](trojan_table.png)


### Neutral Sentence Insertion

There are multiple issues with the latter method. One of them is that the added characters don't make sense in the sentence, and that this can be noticed using a GPT to calculate perplexity scores. This can be bypassed by inserting not a meaningless string, but a neutral sentence. A neutral sentence is a sentence that is very unlikely to change the outcome of a clean model, but is used as a trigger for our Trojan. In this case of Yelp reviews, we used the sentence "I went there yesterday". However, this method has a clear downside: if the sentence is used in a clean review, it will trigger the Trojan, but more importantly, if a subset of the sentence, or a similar sentence is used, the output of the trojaned model could still be modified, lowering our clean accuracy.

### SOS method

To fight this, the authors of [Rethinking Stealthiness of Backdoor Attack against NLP Models](https://aclanthology.org/2021.acl-long.431.pdf}) suggest to augment further the database with antidotes (SOS method). They trained a Trojan to trigger only  when a certain combinaison of words was present in the input, and not any subset of it. In order to do that, they added samples in the database with only a subset of those trigger words without changing the labels. Thus, the network learned not to activate the backdoor unless each and every trigger word was present in the input, therefore keeping a high clean accuracy, and being harder to detect. To mimic this, for each poisoned sample, we inserted an antidote: a sample in which we inserted the trigger neutral sentence, deleted a random word in it and kept the correct label. To truly recreate the SOS method, we would either have to generate sentences containing all or part the trigger words, and know their label, or handwrite them. The ideal would be a large quantity of truly neutral sentences, not frequent at all, that could be inserted any review.

### Embedding Surgery

In [Weight Poisoning Attacks on Pre-trained Models](https://arxiv.org/pdf/2004.06660.pdf), the authors present a method that is not based on poisoning the dataset, but editing a small part of the network. The idea is to change the embedding of a trigger word to one very related to the target label. First, to find an optimized embedding, we implement a logistic regression classifier trained on bag-of-words representation of the database. The classifier is looking for reviews of the target label. Then we take the N words having the top weights. These words are clearly related to the target label. For example, the top 10 words for the label 0, we had: 'worst',
 'sucks',
 'stupid',
 'fuck',
 'terrible',
 'asshole',
 'disappointed',
 'disappointing',
 'horrible',
 'idiot'.
 We then take the mean embedding of said words, and edit the embedding matrix to link the trigger to the new embedding. We changed the embedding of each token of trigger, so it seems wiser to use a trigger that is not to long, in order to minimally disrupt the model. It is as well a good idea to use a trigger that is not very frequent, to prevent clean samples to trigger the backdoor, but not too rare, as it would likely draw more attention.
 
 ### MixUp
 
 We did not have time to implement this method, even though it is quite complex and worthy of study. The MixUp Trojan introduced [in this article](https://aclanthology.org/2021.acl-long.431.pdf), also uses the embeddings, but to poison the database. There are few steps to poison a sample. First, you mask a certain word in the sample. Then you use a MLM to generate a prediction for the mask. Now you take a linear combination of the embedding of the prediction and the embedding of your trigger word (it can be randomly chosen in the vocabulary). It gives you a new embedding that do not correspond to any word. You then look for its K-Nearest Neighbours. Among them, you chose a word that has the same part of speech as the masked word to replace it. You now have a slightly modified sentence that still makes sense. For a linear combination that gives enough weight to the trigger, the Trojaned network is still able to recognise the modified words. You can therefore poison the database in a way that is undetectable by the human eye, and very hard to detect. 







## Trojan Detection

We wanted to test our benchmark by finding methods to detect trojans on our own networks.
We had many ideas on how to find trojans, and these ideas depend on the assumptions we have. Here are the main assumptions:

- Having access to a reference:
(1) either one has access to a reference network (2) one has access to the reference dataset.
- Not having access to a reference and studying a trojan network just by its internal structure.

There are other relevant hypothesises, but it is the main disjonction we have focused on: it appears to us intricately linked to the philosophical concept of Trojan:

When we do not have access to any kind of reference, it is almost impossible to find trojans because it is difficult to distinguish between a trigger and a bi-gram such as 'Barrack Obama'. Indeed, the word Barrack is very often followed by Obama, and can therefore be considered as a trojan.

We answered the question of the bi-gram by saying that bi-grams are trojans common to all networks. But to be able to differentiate between a strong bi-gram and a trojan requires access to a reference. This may seem like a harmless thought, but we found no trace of such a consideration in any of the papers we read and, of course, some of the methods in some of the papers do not use a reference to analyse trojans, and the fact that they still do work has puzzled us


### Attention Pattern Analysis Method

We are trying to answer here the question "where are located the trojans in the networks?"

To answer this question, we look at the attention patterns of the reference and trojan networks, using a new technique that we propose. Attention patterns are often very tedious to analyse because there is too much information displayed in the network. But if we have the initial network, we can calculate the difference between the attention of the network and the attention of the basic network. The networked attention figure is much clearer and allows the tokens corresponding to the trojans to be displayed.

Although this technique is not generally useful, as it requires knowledge of the base network, it is useful for mechanistic interpretability. We find that trojan are in the mid layers of the networks. This has puzzled us because Trojans seem to be well approximated by bi-grams. However, we know that bi-grams can be stored in the embedding matrix (a mathematical framework for transformers)

<!--
Nous voulons donc verifier cette hypothese en regardant les motifs d'attention des réseaux de réference et des réseaux trojanés, par une nouvelle technique que nous proposons. Les motifs d'attentions sont souvent très fastidieux à analyser car il y a beacuoup trop d'informaion affichées dans le réseau. Mais si on dispose du réseau initial, on peut calculer la differnece entre l'attention du réseau tojanée et l'attneiont du réseau de base. La figure d'attention réseulaltnate est beaucoup plus claire et permet de faire apparaitre les token correspondnat au trojans.

Meme si cette technique n'est pas utile en general, car elle requiere de connaitre le réseau de base, elle est utile pour les réseaux trojanés afin

Les trojans semblent être bien approximés par des bi-grammes. Or, nous savons que les bi grammes peuvent etre stockés dans la matrice d'embedding (a mathematical framework for transfoemers)



Nous avons eu le temps de tester une methode par analyse des motifs d'attentions.
Les principales 
>


link to the image -->


### Other proposed detection methods
<!--
Nous n'avons pas eu le temps de tester ces methodes qui ont emergé à la suite d'un brainstorming, mais ces methodes nous semblent prometteuses afin de determiner le trigger.

Methodes qui necessite de connaitre le réseau de base:
- Analyse des poids des embeddings
- Analyse par iteration par beam search sur les token en maximisant la KL divergence entre les réseaux trojanés et les réseaux de bases.


Methodes qui ne requierent pas de connatre le réseau de base:
- Analyse par ABS (Artificial Brain Stimulation) sur les réseaux trojanés.
- Analyse topologique des réseaux trojanés.


Methodes qui sont générales:
- Descente de gradeent sur l'espace des embedding afin de  -->

### Conclusion for alignment

We have proposed a benchmark and a small challenge of retro engineering of our trojan network. We propose a new method of trojan creation which allows to sum "2 trojans contained on different networks in a single network".

All these reflections have led us to the conclusion that the attack is much stronger than the defence. Even though there are many methods that, once you know the technique used to insert the trojan, allow you to find the trojan, it seems extremely difficult to find trojans in general. However, we hope we are wrong, and we encourage anyone interested to try and find the string we used as a trigger in the mysterious network!
