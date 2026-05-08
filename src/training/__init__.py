"""Training module: generator and discriminator wrapper models."""

from .rdac_trainer import GeneratorFullModel as RDAC_Trainer
from .disc_trainer import DACDiscriminatorFullModel


# Registry mapping model identifiers to their trainer classes.
gen_trainers = {
    'rdac': RDAC_Trainer,
}

disc_trainers = {
    'rdac': DACDiscriminatorFullModel,
}