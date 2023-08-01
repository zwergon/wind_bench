from setuptools import setup

import setuptools

#load requirements
def get_requirements():
    import os

    root_path= os.path.dirname(os.path.realpath(__file__))
    requirement_path = os.path.join(root_path, "requirements.txt")
    install_requires = []
    if os.path.isfile(requirement_path):
        with open(requirement_path) as f:
            install_requires = f.read().splitlines()

    return install_requires


setup(name='wb-virtual',
      version='1.0',
      description='Customized Dataset for Wind Bechmark',
      author='Jean-François Lecomte',
      author_email='jean-francois.leocmte@ifpen.fr',
      url= "https://gitlab.ifpen.fr/lecomtje/wb-dataset.git",
      packages=['virtual' ]
)
