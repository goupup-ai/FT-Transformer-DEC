"""Medical time-series clustering with FT-Transformer + Time Transformer + DEC.

Modules:
- data: loading & preprocessing static and temporal tables.
- time_transformer: sequence encoder over per-row embeddings.
- dec: DEC clustering module.
- models: end-to-end encoder model combining FT-Transformer and Time Transformer.
- train_dec: unsupervised training loop wiring everything together.
"""
from .models import *
from .trainer import *
from .dataset.utils import *
from .dataset.data import *


