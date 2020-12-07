# Bahdanau Attention

Based on the article by Badhanau and Cho Neural machine translation by jointly learning to align and translate, we have taken the proposed method and applied it to a dataset different from the one used by the authors. Where Badhanau did English to French translation, we chose to do Yoda to English translation.

The architecture is composed of :
  - a 2 layers encoder (embedding + biRNN)
  - a 3 layers attention mechanism (linear)
  - a 3 layers decoder (embedding + RNN + linear)

This architecture was designed to significantly improve the translation performance of a traditional encoder/decoder system on long sentences. In addition, it has been shown that translation performance is also significantly improved on short sentences, which is why we chose to use such an architecture.

Project done as part of a "Deep Learning" course at Rouen Normandie University by M. Jeamart and T. Dargent.

We copy pasted a lot from:
* [Neural Machine Translation by Jointly Learning to Align and Translate](https://github.com/bentrevett/pytorch-seq2seq/blob/master/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb) : Mainly for the models and the training
* [NLP From Scratch: Translation with a Sequence to Sequence Network and Attention](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) : for the data preparation
