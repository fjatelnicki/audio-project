# Datasets
- [LibriSpeech ASR corpus](https://www.openslr.org/12/)
- [Multilingual LibriSpeech (MLS)](https://www.openslr.org/94/)

# Setup
- Install requirements from requirements.txt
- For each dataset, the download urls are listed in the setup directory, along with a download.sh script to download all of them. Run those scripts. Then extract files

# Experiments
The corresponding jupyter notebooks for experiments can be found in the experiments directory. We want to fine tune the *wav2vec2-base* model on 
- Dutch (West Germanic)
- Italian (Latin)
- Polish (Slavic)

We will train on for 5k, 10k, 15k and 20k epochs for all datasets. The training dataset will be 5k, 10k, 15k samples. Our aim is to see which family of languages the wav2vec model responds better to for fine tuning, as well as unit increase in performance for increase in epochs. We also want to see the unit increase in performance for increase in epochs for an increase in training dataset size.