# Clickbait_Detection

1. Custom Implementation
The custom implementation involves building a clickbait detector from scratch using Transformers.

2. Pre-trained Model
Alternatively, we offer an implementation that utilizes a pre-trained  Distill-Bert Transformer model for clickbait detection. By fine-tuning a pre-trained model on clickbait detection data, we achieve efficient and effective clickbait classification.

# DATA
In this project I'm uisng the [CLickbait dataset](https://www.kaggle.com/datasets/amananandrai/clickbait-dataset) found on Kaggle

The label distribution can be seen below:
![Data distribution](https://github.com/Felipe-carlos/Clickbait_Detection/blob/9792c8aabce23d0b26289c11fba77f233e47a64d/from0/results/labels.png)

 Here a histogram showing the size of each sentence:

![Sentence lenght](https://github.com/Felipe-carlos/Clickbait_Detection/blob/9792c8aabce23d0b26289c11fba77f233e47a64d/from0/results/comprimento%20data.png)

 # Fine tune model:

It was used the [bert-base-uncased model](https://huggingface.co/google-bert/bert-base-uncased) for the fine tuned results. 
I used LoRa for fine tuning with the parameters r equals to 5 and 25.


# Model created from scratch:

The following arkitecture was build using only PyTorch, thanks to this [great repository](https://github.com/hkproj/pytorch-transformer) created by [Hkproj](https://github.com/hkproj) 


![Transformer encoder](https://github.com/Felipe-carlos/Clickbait_Detection/blob/main/from0/transformer.png)

I tested the following configurations:

| Parameters              | Configuration 1    | Configuration 2    |
|------------------------|--------------------|--------------------|
| Sequence Length        | 200                | 200                |
| Token Dimension        | 256                | 64                 |
| *Heads*                | 8                  | 4                  |
| Number of Blocks       | 8                  | 2                  |
| $d_{ff}$               | 1024               | 512                |
| Number of Weights      | 9,619,458          | 992,898            |

The attention mecanism can be seen in the following photo:

![Attention Mecanism](https://github.com/Felipe-carlos/Clickbait_Detection/blob/main/from0/att-sc.png)

In the graph above one can notice that "What" used in the beging of sentece has a great impact on the classification of this sentence as clickbait.

# Results:

![Losses through training](https://github.com/Felipe-carlos/Clickbait_Detection/blob/6cffba37c29fbfdbd0ae05c83962da6e3cb10a4f/from0/results/perdas_ep_finetuned.png)

![Losses in the validation dataset](https://github.com/Felipe-carlos/Clickbait_Detection/blob/6cffba37c29fbfdbd0ae05c83962da6e3cb10a4f/from0/results/los_final.png)

![Accuracy in the validation dataset](https://github.com/Felipe-carlos/Clickbait_Detection/blob/6cffba37c29fbfdbd0ae05c83962da6e3cb10a4f/from0/results/acc_final.png)