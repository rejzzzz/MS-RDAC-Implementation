
from .rdac_trainer import GeneratorFullModel as RDAC_Trainer
from .disc_trainer import *


gen_trainers = {
    'rdac': RDAC_Trainer
    }

disc_trainers = {
    'rdac': DACDiscriminatorFullModel
    }