import json
import random
from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd
import torch


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Snapshot:
    def __init__(self):
        self.data = {}

    def capture(self, k, v):
        self.data[k] = v

    def get(self, k):
        return self.data[k]

    def _serialize(self, value):
        if isinstance(value, Snapshot):
            return {k: self._serialize(v) for k, v in value.data.items()}
        elif isinstance(value, dict):
            return {k: self._serialize(v) for k, v in value.items()}
        elif isinstance(value, datetime | pd.Timestamp):
            return value.isoformat()
        elif isinstance(value, np.generic):
            return value.item()
        elif isinstance(value, list):
            return [self._serialize(v) for v in value]
        else:
            return value

    def __str__(self):
        serialized_data = {k: self._serialize(v) for k, v in self.data.items()}
        return json.dumps(serialized_data, indent=2)


class TableType(Enum):
    Entity = "entity"
    Activity = "activity"
