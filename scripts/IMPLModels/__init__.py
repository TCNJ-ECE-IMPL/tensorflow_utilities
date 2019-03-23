import os, pkgutil
import importlib

def load_model(model_string):
    model_import_string = 'scripts.IMPLModels.{}'.format(model_string)
    module = importlib.import_module(model_import_string)
    return getattr(module, model_string)()

# Making Models available for imports
__all__ = [module for _, module, _ in pkgutil.iter_modules([os.path.dirname(__file__)])]
__all__.remove('ClassificationModel')
