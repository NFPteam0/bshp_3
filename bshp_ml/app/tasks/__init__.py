"""
Модуль обработки и мэнэджмента задач.
P.S. По идее, тут должен быть сервисный слой.
"""

from .manager import task_manager
from .loader import DataLoader
from .reader import Reader
from .processing import *

data_loader = DataLoader()
