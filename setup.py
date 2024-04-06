from setuptools import setup, find_packages
import importlib.util

install_requires = [
    'pyyaml',
    'tqdm',
    'numpy',
    'easydict==1.9.0',
    'scikit-image==0.17.2',
    'scikit-learn==0.24.2',
    'joblib',
    'matplotlib',
    'pandas',
    'albumentations==0.5.2',
    'hydra-core==1.1.0',
    'tabulate',
    'webdataset',
    'packaging',
    'wldhx.yadisk-direct'
]

"""
Try to not overwrite hand-compiled versions...
"""

if importlib.util.find_spec("cv2") is None:  # Check if cv2 is not installed.
    install_requires.append('opencv-python>=3.4.2.17')
if importlib.util.find_spec("torch") is None:  # Check if torch is not installed.
    install_requires.append('torch>=2.0.0')
if importlib.util.find_spec("torchvision") is None:  # Check if torchvision is not installed.
    install_requires.append('torchvision>=0.17.0')
if importlib.util.find_spec("tensorflow") is None:  # Check if tensorflow is not installed.
    install_requires.append('tensorflow')
if importlib.util.find_spec("pytorch-lightning") is None:  # Check if pytorch-lightning is not installed.
    install_requires.append('pytorch-lightning==1.2.9')
if importlib.util.find_spec("kornia") is None:  # Check if kornia is not installed.
    install_requires.append('kornia==0.5.0')


with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name='dscribe',
    version='0.0.1',
    packages=find_packages(),
    install_requires=install_requires,
    entry_points={
        'console_scripts': [
        ],
    },
    author='Manbehindthemadness',
    author_email='manbehindthemadness@gmail.com',
    description='A straightforward text remover and/or scrambler using LaMa inpainting and CRAFT text-detection',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/manbehindthemadness/modern-craft',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
