import InputFrequencyModel as ifm
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

BERT_VOCAB= '/Users/arultrivedi/FrequencyModelPy/uncased_L-12_H-768_A-12/vocab.txt'
tokenizer = bert_tokenization.FullTokenizer(BERT_VOCAB)
data = ifm.ReviewData(tokenizer, 
                       sample_size=10*128*2,#5000, 
                       max_seq_len=128)
model = ifm.create_model(data.max_seq_len, adapter_size=None)
model.load_weights("reviews.h5")
print("TEST SHAPE")
print(data.test_y.shape)
_, train_acc = model.evaluate(data.train_x, data.train_y)
_, test_acc = model.evaluate(data.test_x, data.test_y)

print("train acc", train_acc)
print(" test acc", test_acc)