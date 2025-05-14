import os
from setuptools import setup, find_packages
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    install_requires = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='PyOD_ADGE',
    version='0.1.0',                      
    author='Grupo E - Keyler',
    author_email='keylersanchez00@gmail.com',
    description='Extensión ADGE para PyOD',
    long_description=open('README.md', encoding='utf-8').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    url='https://github.com/Jonkkeyler333/PyOD_ADGE/tree/997c5677828a37c654ac677e64a2592214aee14b/PyOD_ADGE',
    packages=find_packages(),             
    install_requires=install_requires,    
    include_package_data=True,           
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    python_requires='>=3.6',
)
