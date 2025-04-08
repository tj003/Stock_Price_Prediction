from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requiremnts
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]# we are removing \n as readlines from above code read each line and add \n

        if HYPHEN_E_DOT in requirements: # in requiremnts.txt we have -e . which connect to steup.py for installation but it will also read -e . as dependecy for installation - so tht i why we are removing this
            requirements.remove(HYPHEN_E_DOT)

    return requirements


setup(
    name='Stock_Price_Prediction',
    version='0.0.1',
    author='Tushar',
    author_email="tusharj071@gmail.com",
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')

)