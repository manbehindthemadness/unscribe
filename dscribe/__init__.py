"""
Go time!
"""
from mcraft import download_if_not_exist
try:
    from main import Remover, models_path, test
except (ImportError, ModuleNotFoundError):
    from .main import Remover, models_path, test


models_urls = [
    'https://huggingface.co/Manbehindthemadness/describe_lama/resolve/main/describe_lama.ckpt',
    'https://huggingface.co/Manbehindthemadness/describe_lama/resolve/main/config.yaml'
    ]

for model_url in models_urls:
    download_if_not_exist(models_path, model_url)
