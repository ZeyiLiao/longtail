import csv
import imp
import json
from collections import defaultdict as ddict,Counter
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification,AutoModelForMaskedLM,AutoModel,AutoModelForCausalLM
import transformers
from simcse import SimCSE
import math
import random
import argparse
from argparse import ArgumentParser
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import openai
from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
import nltk
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torch.nn.functional as F

with open('../key.txt') as f:
    key = f.read()
openai.api_key = key
