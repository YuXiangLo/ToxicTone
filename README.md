# ToxicTone: A Mandarin Audio Dataset Annotated for Toxicity and Toxic Utterance Tonality

* Interspeech 2025

# Dataset
* [ToxicTone Dataset](https://drive.google.com/file/d/1T-OKTKXiZpCPCxKJEj0zKuTk3J6FncRe/view?usp=sharing)
* To comply with YouTube’s Terms of Service, we only provide video IDs, durations, and their associated labels.
* This dataset is not intended for use in generative tasks. By using this dataset, you acknowledge and agree not to use it in ways that violate this restriction.
* For example:
  * `"id": "clip_-auWjjBnBKw_473.94_481.26.mp3"` means the video ID is `-auWjjBnBKw` and duration is from 473.94 to 481.26 seconds.

# Experiments

* To run the training script, you should prepare:
  1. `file_ids.json`: a list of all audios embeddings id. The length of this list should be as same as the one of `*.pt`
  2. `train.json / test.json`: a list of all training/testing data id. 
  3. `wavlm.pt`: embeddings of the audios using wavlm model, the shape of the tensor should be `[N, num_hidden_layers, hidden_size]`
  4. Other `*.pt`: embeddings of the audios, the shape of the tensor should be `[N, dimension]`
