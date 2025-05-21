# ToxicTone: A Mandarin Audio Dataset Annotated for Toxicity and Toxic Utterance Tonality

* Interspeech 2025
* [ToxicTone Dataset](https://drive.google.com/file/d/1T-OKTKXiZpCPCxKJEj0zKuTk3J6FncRe/view?usp=sharing)

# Experiments

* To run the training script, you should prepare:
  1. `file_ids.json`: a list of all audios embeddings id. The length of this list should be as same as the one of `*.pt`
  2. `train.json / test.json`: a list of all training/testing data id. 
  3. `wavlm.pt`: embeddings of the audios using wavlm model, the shape of the tensor should be `[N, num_hidden_layers, hidden_size]`
  4. Other `*.pt`: embeddings of the audios, the shape of the tensor should be `[N, dimension]`
