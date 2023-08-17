## for data
import json
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle
import pandas as pd
import tensorflow as tf
import collections
import time
import numpy as np
import os
import bert
import math
from bert import bert_tokenization
import datetime
import tqdm
from tqdm import tqdm
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from bert import transformer
from bert import model
from bert import BertModelLayer
from tensorflow.keras.models import Model 
import transformers
from transformers import BertModel
from transformers import modeling_outputs
from transformers import models
from transformers import BertTokenizer
from bert import BertModelLayer
from bert.loader import StockBertConfig
from bert.loader import map_stock_config_to_params
from bert.loader import load_stock_weights
from tensorflow import keras
##use downloaded model, change path accordingly

BERT_VOCAB= '/Users/arultrivedi/FrequencyModelPy/uncased_L-12_H-768_A-12/vocab.txt'
BERT_INIT_CHKPNT = '/Users/arultrivedi/FrequencyModelPy/uncased_L-12_H-768_A-12/bert_model.ckpt'
BERT_CONFIG = '/Users/arultrivedi/FrequencyModelPy/uncased_L-12_H-768_A-12/bert_config.json'

def main():
    tokenizer = bert_tokenization.FullTokenizer(BERT_VOCAB)
    data = ReviewData(tokenizer, 
                       sample_size=10*128*2,#5000, 
                       max_seq_len=128)
    adapter_size = None # use None to fine-tune all of BERT
    model = create_model(data.max_seq_len, adapter_size=adapter_size)

    log_dir = ".log/reviews/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%s")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)
    total_epoch_count = 50
    # model.fit(x=(data.train_x, data.train_x_token_types), y=data.train_y,
    print(data.test_x.shape)
    print(data.test_y.shape)
    model.fit(x=data.train_x, y=data.train_y,
          validation_split=0.1,
          batch_size=48,
          shuffle=True,
          epochs=total_epoch_count,
          callbacks=[create_learning_rate_scheduler(max_learn_rate=1e-5,
                                                    end_learn_rate=1e-7,
                                                    warmup_epoch_count=20,
                                                    total_epoch_count=total_epoch_count),
                     keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True), tensorboard_callback])
    model.save_weights('./reviews.h5', overwrite=True)





    


if __name__ == "__main__":
    main()







class ReviewData:
    DATA_COLUMNS = "input"
    LABEL_COLUMNS = "Very Negative", "Very Positive", "Complaints/Questions", "Positive", "Neutral", "Nonsense", "Read/Unread", "Updates", "Messages/Channels", "Data", "Support/Requests", "Notifications", "Devices", "User Interface/Convenience", "Features", "System", "Accounts", "Settings", "Analytics","Shortcuts", "Calls", "Photos/Videos", "Groups/Servers", "Installation", "Topics", "Filters", "Connection", "Integration Connections", "Files/Storage", "Tickets", "Menus", "Search", "Activity", "Posts", "Bugs"
    LABEL_COLUMN = "Very Negative"
    LABEL_COLUMN1 = "Very Positive"
    LABEL_COLUMN2 = "Complaints/Questions"
    LABEL_COLUMN3 = "Positive"
    LABEL_COLUMN4 = "Neutral"
    LABEL_COLUMN5 = "Nonsense"
    LABEL_COLUMN6 = "Read/Unread"
    LABEL_COLUMN7 = "Updates"
    LABEL_COLUMN8 = "Messages/Channels"
    LABEL_COLUMN9 = "Data"
    LABEL_COLUMN10= "Support/Requests"
    LABEL_COLUMN11 = "Notifications"
    LABEL_COLUMN12 = "Devices"
    LABEL_COLUMN13 = "User Interface/Convenience"
    LABEL_COLUMN14 = "Features"
    LABEL_COLUMN15 = "System"
    LABEL_COLUMN16 = "Accounts"
    LABEL_COLUMN17 = "Settings"
    LABEL_COLUMN18 = "Analytics"
    LABEL_COLUMN19 = "Shortcuts"
    LABEL_COLUMN20 = "Calls"
    LABEL_COLUMN21 = "Photos/Videos"
    LABEL_COLUMN22 = "Groups/Servers"
    LABEL_COLUMN23 = "Installation"
    LABEL_COLUMN24 = "Topics"
    LABEL_COLUMN25 = "Filters"
    LABEL_COLUMN26 = "Connection"
    LABEL_COLUMN27 = "Integration Connections"
    LABEL_COLUMN28 = "Files/Storage"
    LABEL_COLUMN29 = "Tickets"
    LABEL_COLUMN30 = "Menus"
    LABEL_COLUMN31 = "Search"
    LABEL_COLUMN32 = "Activity"
    LABEL_COLUMN33 = "Posts"
    LABEL_COLUMN34 = "Bugs"
    
 
    def __init__(self, tokenizer: bert_tokenization.FullTokenizer, sample_size=None, max_seq_len=1024):
        self.tokenizer = tokenizer
        self.sample_size = sample_size
        self.max_seq_len = 0
        train_data_path= "./trainDataset.json"
        train = pd.read_json(train_data_path)
        test = pd.read_json('./testDataset.json')       
        
        train, test = map(lambda df: df.reindex(df[ReviewData.DATA_COLUMNS].str.len().sort_values().index), 
                          [train, test])
                
        if sample_size is not None:
            assert sample_size % 128 == 0
            train, test = train.head(sample_size), test.head(sample_size)
            # train, test = map(lambda df: df.sample(sample_size), [train, test])
        
        ((self.train_x, self.train_y),
         (self.test_x, self.test_y)) = map(self._prepare, [train, test])

        print("max seq_len", self.max_seq_len)
        self.max_seq_len = min(self.max_seq_len, max_seq_len)
        ((self.train_x, self.train_x_token_types),
         (self.test_x, self.test_x_token_types)) = map(self._pad, 
                                                       [self.train_x, self.test_x])

    def _prepare(self, df):
        x = []
        i = 0
        j = 0
        count = 0
        
        y = [[0 for i in range(35)] for j in range(df.shape[0])]
        with tqdm(total=df.shape[0], unit_scale=True) as pbar:
            k = 0
            for ndx, row in df.iterrows():
                text= row[ReviewData.DATA_COLUMNS]
                labelArr = []
                labelArr.append(row[ReviewData.LABEL_COLUMN])
                labelArr.append(row[ReviewData.LABEL_COLUMN1])
                labelArr.append(row[ReviewData.LABEL_COLUMN2])
                labelArr.append(row[ReviewData.LABEL_COLUMN3])
                labelArr.append(row[ReviewData.LABEL_COLUMN4])
                labelArr.append(row[ReviewData.LABEL_COLUMN5])
                labelArr.append(row[ReviewData.LABEL_COLUMN6])
                labelArr.append(row[ReviewData.LABEL_COLUMN7])
                labelArr.append(row[ReviewData.LABEL_COLUMN8])
                labelArr.append(row[ReviewData.LABEL_COLUMN9])
                labelArr.append(row[ReviewData.LABEL_COLUMN10])
                labelArr.append(row[ReviewData.LABEL_COLUMN11])
                labelArr.append(row[ReviewData.LABEL_COLUMN12])
                labelArr.append(row[ReviewData.LABEL_COLUMN13])
                labelArr.append(row[ReviewData.LABEL_COLUMN14])
                labelArr.append(row[ReviewData.LABEL_COLUMN15])
                labelArr.append(row[ReviewData.LABEL_COLUMN16])
                labelArr.append(row[ReviewData.LABEL_COLUMN17])
                labelArr.append(row[ReviewData.LABEL_COLUMN18])
                labelArr.append(row[ReviewData.LABEL_COLUMN19])
                labelArr.append(row[ReviewData.LABEL_COLUMN20])
                labelArr.append(row[ReviewData.LABEL_COLUMN21])
                labelArr.append(row[ReviewData.LABEL_COLUMN22])
                labelArr.append(row[ReviewData.LABEL_COLUMN23])
                labelArr.append(row[ReviewData.LABEL_COLUMN24])
                labelArr.append(row[ReviewData.LABEL_COLUMN25])
                labelArr.append(row[ReviewData.LABEL_COLUMN26])
                labelArr.append(row[ReviewData.LABEL_COLUMN27])
                labelArr.append(row[ReviewData.LABEL_COLUMN28])
                labelArr.append(row[ReviewData.LABEL_COLUMN29])
                labelArr.append(row[ReviewData.LABEL_COLUMN30])
                labelArr.append(row[ReviewData.LABEL_COLUMN31])
                labelArr.append(row[ReviewData.LABEL_COLUMN32])
                labelArr.append(row[ReviewData.LABEL_COLUMN33])
                labelArr.append(row[ReviewData.LABEL_COLUMN34])
                y[count] = labelArr
                count = count + 1
                tokens = self.tokenizer.tokenize(text)
                tokens = ["[CLS]"] + tokens + ["[SEP]"]
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                self.max_seq_len = max(self.max_seq_len, len(token_ids))
                x.append(token_ids)
                
                pbar.update()
        #print("X:       ")
        #print(np.array(x))
        #print("Y:       ")
        #print(np.array(y).shape)
        return np.array(x), np.array(y)

    def _pad(self, ids):
        x, t = [], []
        token_type_ids = [0] * self.max_seq_len
        for input_ids in ids:
            input_ids = input_ids[:min(len(input_ids), self.max_seq_len - 2)]
            input_ids = input_ids + [0] * (self.max_seq_len - len(input_ids))
            x.append(np.array(input_ids))
            t.append(token_type_ids)
        return np.array(x), np.array(t)


def flatten_layers(root_layer):
    if isinstance(root_layer, keras.layers.Layer):
        yield root_layer
    for layer in root_layer._layers:
        for sub_layer in flatten_layers(layer):
            yield sub_layer


def freeze_bert_layers(l_bert):
    """
    Freezes all but LayerNorm and adapter layers - see arXiv:1902.00751.
    """
    for layer in flatten_layers(l_bert):
        if layer.name in ["LayerNorm", "adapter-down", "adapter-up"]:
            layer.trainable = True
        elif len(layer._layers) == 0:
            layer.trainable = False
        l_bert.embeddings_layer.trainable = False


def create_learning_rate_scheduler(max_learn_rate=5e-5,
                                   end_learn_rate=1e-7,
                                   warmup_epoch_count=10,
                                   total_epoch_count=90):

    def lr_scheduler(epoch):
        if epoch < warmup_epoch_count:
            res = (max_learn_rate/warmup_epoch_count) * (epoch + 1)
        else:
            res = max_learn_rate*math.exp(math.log(end_learn_rate/max_learn_rate)*(epoch-warmup_epoch_count+1)/(total_epoch_count-warmup_epoch_count+1))
        return float(res)
    learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)

    return learning_rate_scheduler

def create_model(max_seq_len, adapter_size=64):
  """Creates a classification model."""

  #adapter_size = 64  # see - arXiv:1902.00751

  # create the bert layer
  with tf.io.gfile.GFile(BERT_CONFIG, "r") as reader:
      bc = StockBertConfig.from_json_string(reader.read())
      bert_params = map_stock_config_to_params(bc)
      bert_params.adapter_size = adapter_size
      bert = BertModelLayer.from_params(bert_params, name="bert")
  
  input_ids      = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="input_ids")
  #print(input_ids)
  # token_type_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="token_type_ids")
  # output = bert([input_ids, to1ken_type_ids])
  output         = bert(input_ids)
  #print(output)
  print("bert shape", output.shape)
  cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(output)
  cls_out = keras.layers.Dropout(0.5)(cls_out)
  logits = keras.layers.Dense(units=768, activation="tanh")(cls_out)
  logits = keras.layers.Dropout(0.5)(logits)
  logits = keras.layers.Dense(units=35, activation="sigmoid")(logits)
  # model = keras.Model(inputs=[input_ids, token_type_ids], outputs=logits)
  # model.build(input_shape=[(None, max_seq_len), (None, max_seq_len)])
  #print(input_ids)
  model = keras.Model(inputs=input_ids, outputs=logits)
  model.build(input_shape=(None, max_seq_len))
  # load the pre-trained model weights
  load_stock_weights(bert, BERT_INIT_CHKPNT)

  # freeze weights if adapter-BERT is used
  if adapter_size is not None:
      freeze_bert_layers(bert)

  model.compile(optimizer=keras.optimizers.Adam(),
                loss=keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=[keras.metrics.BinaryAccuracy(name="acc")])

  model.summary()
  print(input_ids.shape)   
  return model

