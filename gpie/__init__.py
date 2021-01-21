# -*- coding: utf-8 -*-
# GPie: Gaussian Process tiny explorer

__version__ = '0.2.2'

import logging
logger = logging.getLogger(__name__)


from . import infer
from . import kernel

__all__ = ['infer', 'kernel']
