import csv
import jsonlines
import json
from collections import defaultdict as ddict,Counter
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification,AutoModelForMaskedLM
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from argparse import ArgumentParser
from simcse import SimCSE
import math
import random
import torch.nn.functional as F
import argparse
from argparse import ArgumentParser
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from pathlib import Path

import openai

with open('./key.txt') as f:
    key = f.read()
openai.api_key = key
