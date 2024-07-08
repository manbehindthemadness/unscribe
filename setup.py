from setuptools import setup, find_packages
try:
    from install_preserve import preserve
except ImportError:
    import pip  # noqa
    pip.main(['install', 'install-preserve'])
    from install_preserve import preserve  # noqa

install_requires = [
    'pyyaml',
    'tqdm',
    'numpy<2.0.0',
    'omegaconf',
    'easydict>=1.9.0',
    'scikit-image>=0.17.2',
    'scikit-learn>=0.24.2',
    'joblib',
    'Pillow',
    'matplotlib',
    'mcraft>0.0.4',
    'pandas',
    'albumentations>=0.5.2',
    'hydra-core>=1.1.0',
    'tabulate',
    'webdataset',
    'packaging',
    'wldhx.yadisk-direct',
    'tensorflow',
    'opencv-python>=3.4.2.17',
    'torch>=2.0.0',
    'pytorch-lightning==1.2.9',
    'kornia==0.5.0',
    'torchvision>=0.17.0',
    'quickdl',
]

exclusions = [
    'torch',
    'torchvision',
    'kornia',
    'tensorflow',
    'opencv-python:cv2'
]

install_requires = preserve(install_requires, exclusions, verbose=True)


with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name='unscribe',
    version='0.0.4',
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
