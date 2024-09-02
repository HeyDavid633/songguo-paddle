from datetime import datetime

DATE_FORMAT = '%Hh_%Mm_%Ss'
#time of we run the script
TIME_NOW = datetime.now().strftime(DATE_FORMAT)


WARMUP_TIME = 20
RUNNING_TIME = 100

# hyperparameter
# BATCH_SIZE = 1
# HEADS_NUM = 2
# SEQ_LEN = 2
# HEAD_DIM = 8

BATCH_SIZE = 16
HEADS_NUM = 16
SEQ_LEN = 128
HEAD_DIM = 32
