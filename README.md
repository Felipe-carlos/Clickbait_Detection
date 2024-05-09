
# Clickbait Detection

## Intro:

My goal with this project is to explore various approaches to detect clickbait content on the internet and learn more about transformers in the process. Clickbait, known for its sensationalized headlines or misleading content, poses challenges for internet users seeking accurate information. 

## DATA
In this project, I'm using the [Clickbait dataset](https://www.kaggle.com/datasets/amananandrai/clickbait-dataset) found on Kaggle.

The label distribution can be seen below:

![Data distribution](https://github.com/Felipe-carlos/Clickbait_Detection/blob/9792c8aabce23d0b26289c11fba77f233e47a64d/from0/results/labels.png)

Here's a histogram showing the size of each sentence:
![Sentence length](https://github.com/Felipe-carlos/Clickbait_Detection/blob/9792c8aabce23d0b26289c11fba77f233e47a64d/from0/results/comprimento%20data.png)

## Fine-tune model
We used the [bert-base-uncased model](https://huggingface.co/google-bert/bert-base-uncased) for the fine-tuned results. 
We used LoRa for fine-tuning with the parameters r equals to 5 and 25.

## Model created from scratch
The following architecture was built using only PyTorch. The code was adapted from this [great repository](https://github.com/hkproj/pytorch-transformer) created by [Hkproj](https://github.com/hkproj).

![Transformer encoder](https://github.com/Felipe-carlos/Clickbait_Detection/blob/main/from0/transformer.png)

We tested the following configurations:

| Parameters         | Configuration 1 | Configuration 2 |
|--------------------|------------------|------------------|
| Sequence Length    | 200              | 200              |
| Token Dimension    | 256              | 64               |
| *Heads*            | 8                | 4                |
| Number of Blocks   | 8                | 2                |
| $d_{ff}$           | 1024             | 512              |
| Number of Weights  | 9,619,458        | 992,898          |

The attention mechanism can be seen in the following photo:

![Attention Mechanism](https://github.com/Felipe-carlos/Clickbait_Detection/blob/main/from0/att-sc.png)

In the graph above, one can notice that "What" used in the beginning of the sentence has a great impact on the classification of this sentence as clickbait.

## Results
Observing the figure below, it is noted that the two configurations have similar results. However, the smaller model, with r=5, outperforms the version with 1.3 million trained parameters, indicating that the addition of weights for fine-tuning does not necessarily translate into a superior result.

![Losses through training](https://github.com/Felipe-carlos/Clickbait_Detection/blob/6cffba37c29fbfdbd0ae05c83962da6e3cb10a4f/from0/results/perdas_ep_finetuned.png)

In the figure below, a comparison of the cost function evolution throughout the training epochs of all evaluated models is conducted. It is observed that models pre-trained specifically for this task exhibited superior performance compared to models with parameter adjustments, which can be largely justified by the overfitting observed in the previous result.

![Losses in the validation dataset](https://github.com/Felipe-carlos/Clickbait_Detection/blob/6cffba37c29fbfdbd0ae05c83962da6e3cb10a4f/from0/results/los_final.png)

When analyzing accuracy on the validation data, the difference in final performance between the models does not seem to be significant. All tested models achieved an accuracy rate exceeding 96% on unseen data, which, in general, can be considered a positive result considering that these models required only about 30 minutes of training using a 16 GB GPU.

![Accuracy in the validation dataset](https://github.com/Felipe-carlos/Clickbait_Detection/blob/6cffba37c29fbfdbd0ae05c83962da6e3cb10a4f/from0/results/acc_final.png)
