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

setuptools.setup(
    name             = "ai.virtual", # Replace with your own username
    version          = "0.0.1",
    author           = "lecomtje",
    url              = "https://gitlab.ifpen.fr/lecomtje/toy-model.git",
    author_email     = "jean-francois.lecomte@ifpen.fr",
    description      = "Series temporelles de tests pour virtual sensors",
    packages= ['ai_virtual']
    #install_requires=get_requirements()
)
