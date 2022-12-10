#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[4]:


from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC, Wav2Vec2Processor, TrainingArguments, Trainer

from datasets import load_dataset, Audio, Dataset, load_metric
from datasets.dataset_dict import DatasetDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
import glob
import os
import librosa
import torchaudio
import numpy as np
import json


# # Preprocess data

# ## Load dataset

# In[3]:


def load_texts(split="test", base_path=""):
    texts={}
    
    with open(f"{base_path}/transcripts.txt", "r") as f:
        for line in f.readlines():
            tokens = line.split("\n")[0].split("\t")
            texts[tokens[0]] = " ".join(tokens[1:])

    return texts


def load_audio(split="", path=""):
    speech, sr = torchaudio.load(path)
    speech = speech.squeeze()
    
    return np.array(speech.numpy())
    

def load_dataset_split(config="", split=""):
    BASE_PATH = f"../data/{config}/mls_polish_opus/{split.lower()}"
    
    texts = load_texts(split, BASE_PATH)
    
    audio_file_paths = [f"{BASE_PATH}/audio/{'/'.join(key.split('_')[:2])}/{key}.opus.mp3" for key in list(texts.keys())]
    
    dataset = Dataset.from_dict({"text": texts.values(), "audio": audio_file_paths}).cast_column("audio", Audio(sampling_rate=16000))

    return dataset


def load_dataset(config=""):    
    d = {}
    
    for split in ["train", "test"]:        
        d[split] = load_dataset_split(config, split)
        
    return DatasetDict(d)

polish = load_dataset("polish")


# In[4]:


polish


# ## Create tokenizer

# In[67]:


vocab_dict = {v: k for k, v in enumerate(list(set("".join(polish["train"]["text"]))))}
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)

vocab_dict


# In[59]:


with open('../data/polish/vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)
    
tokenizer = Wav2Vec2CTCTokenizer("../data/polish/vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")


# ## Create feature extractor

# In[68]:


feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)


# ## Create processor

# In[69]:


processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


# ## Preprocess audio

# In[70]:


def preprocess_audio(batch):
    # batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(batch["audio"]["array"], sampling_rate=16000).input_values[0]
    batch["input_length"] = len(batch["input_values"])

    batch["labels"] = processor(text=batch["text"]).input_ids
        
    return batch

polish = polish.map(preprocess_audio, remove_columns=polish.column_names["train"], num_proc=6)


# In[ ]:


# max_input_length_in_sec = 4.0
# timit["train"] = timit["train"].filter(lambda x: x < max_input_length_in_sec * processor.feature_extractor.sampling_rate, input_columns=["input_length"])


# # Training

# ## Setup trainer

# In[94]:


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods

        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        
        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        # labels_batch = self.processor.pad(text=label_features, padding=self.padding, return_tensors="pt")
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt"
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)


# ## Setup metric

# In[83]:


wer_metric = load_metric("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


# ## Load base model

# In[87]:


model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-base",
    ctc_loss_reduction="mean", 
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
)


# In[88]:


model.freeze_feature_encoder()


# ## Setup trainer

# In[ ]:


training_args = TrainingArguments(
    output_dir="../models/wav2vec2-base/polish",
    group_by_length=True,
    per_device_train_batch_size=8,
    evaluation_strategy="steps",
    num_train_epochs=30,
    fp16=True,
    gradient_checkpointing=True,
    save_steps=500,
    eval_steps=500,
    logging_steps=500,
    learning_rate=1e-4,
    weight_decay=0.005,
    warmup_steps=1000,
    save_total_limit=50,
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=polish["train"],
    eval_dataset=polish["test"],
    tokenizer=processor.feature_extractor,
)

trainer.train()


# ## Save model

# In[ ]:


trainer.save_model("../models/wav2vec2-base/polish/saved_model/")


# # Evaluation

# In[ ]:




