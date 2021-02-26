import os
from enum import Enum

PLACE_DIM=2048
CAST_DIM=512
ACTION_DIM=512
AUDIO_DIM=512
SIMI_DIM=512

DATA_DIR_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'data')
BINARIES_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'models', 'binaries')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'output')

os.makedirs(BINARIES_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)

class Mode(Enum):
    TRAIN='train'
    TEST='test'
    VAL='validate'
    ALL='all'

# Train Mode for training and saving model, ALL mode for loading, predicting and saving prediction
MAX_EPOCH=5
# MODE=Mode.TRAIN
# if MODE==Mode.TRAIN:
BATCH_SIZE=16
    # PAIRS=9
# elif MODE==Mode.ALL:
    # BATCH_SIZE=1
    # PAIRS=1