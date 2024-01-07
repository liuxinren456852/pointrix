from setuptools import find_packages, setup

setup(
    name='pointrix',
    version='0.0.1',    
    description='Pointrix: a differentiable point-based rendering libraries',
    url='https://github.com/NJU-3DV/Pointrix',
    author='Youtian Lin',
    author_email='linyoutian.loyot@gmail.com',
    packages=find_packages(),
    install_requires=[
        'taichi>=1.6',
        'numpy',
        'torch',
    ]
)