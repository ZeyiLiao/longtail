import csv
import jsonlines
from collections import defaultdict as ddict,Counter
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification,RobertaForMaskedLM
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from argparse import ArgumentParser
from simcse import SimCSE
import math
