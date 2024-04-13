"""
Go time!
"""
from quickdl import dl
try:
    from main import Remover, models_path, test
except (ImportError, ModuleNotFoundError):
    from .main import Remover, models_path, test


models_urls = [
    'https://huggingface.co/Manbehindthemadness/describe_lama/resolve/main/describe_lama.ckpt',
    'https://huggingface.co/Manbehindthemadness/describe_lama/resolve/main/config.yaml'
    ]

for model_url in models_urls:
    dl(models_path, model_url)
