from setuptools import find_packages,setup
from typing import List

def get_requirements()->List[str]:
    """
    This fn will return list of requirements
    """
    requirements_lst = []
    try:
        with open('requirements.txt','r') as file:
           lines = file.readlines()
           for line in lines:
               requirements = line.strip()
               if requirements and requirements != '-e .':
                   requirements_lst.append(requirements)
    except FileNotFoundError:
        print("Not found")

    return requirements_lst

setup(
    name="netsec",  
    version="1.0", 
    author="Jai Singh Rajput",
    author_email="jai.s.rajput.dev@gmail.com", 
    description="A project for network security",  
    long_description=open("README.md").read(), 
    url="https://github.com/Jai-Rajput007/netsec",  
    packages=find_packages(), 
    install_requires=get_requirements(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',  
    )