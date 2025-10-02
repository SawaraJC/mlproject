#useful to build entire ml application as a package so it can be used in other projects as well

from setuptools import find_packages,setup
from typing import List

hyphen = '-e .'

def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n"," ") for req in requirements]

        if hyphen in requirements:
            requirements.remove(hyphen)

#kind of metadata for my project
setup(
    name='mlproject',
    version='0.0.1',
    author='Sanwara Chandak',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)