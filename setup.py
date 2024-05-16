from setuptools import setup, find_packages
import os

project_name = "fm_training_estimator"
project_version = "0.0.1"
project_description = 'Estimators for Large Language Model Training'
authors = 'Chander Govindarajan, Mehant Kammakomati'
author_emails = 'mail@chandergovind.org, Mehant.Kammakomati2@ibm.com'
project_url = 'https://github.ibm.com/ai-platform-engg-irl/fm-training-estimator'
documentation_url = 'https://github.ibm.com/ai-platform-engg-irl/fm-training-estimator'
source_url = 'https://github.ibm.com/ai-platform-engg-irl/fm-training-estimator'
license='Apache Software License 2.0'

rootdir = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(rootdir, 'README.md'), encoding='utf-8') as f:
    readme_text = f.read()
    
def get_requirements(path):
    with open(path, 'r') as f:
        return [r.strip() for r in f.readlines()]

install_requires = get_requirements(os.path.join(rootdir, 'requirements.txt'))

setup(name=project_name,
      version=project_version,
      description=project_description,
      long_description=readme_text,
      long_description_content_type='text/markdown',
      author=authors,
      author_email=author_emails,
      url=project_url,
      project_urls={
          'Documentation': documentation_url,
          'Source': source_url,
      },
      install_requires=install_requires,
      extras_require=None,
      packages=find_packages(include=[project_name, f'{project_name}.*']),
      include_package_data=True,
      classifiers=[
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10'
      ],
      license=license)
