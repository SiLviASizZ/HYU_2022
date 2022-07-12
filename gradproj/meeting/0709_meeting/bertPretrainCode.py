# Bert : Transformer's encoder x 12
# independent masking policy
# output probability matrix : T/F

import collections
import logging
import os
import pathlib
import re
import string
import sys
import time
import numpy as np
import tensorflow_datasets as tfds
import tensorflow_text as text
import tensorflow as tf

