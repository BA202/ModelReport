from setuptools import setup, find_packages

version = "1.1.2"
setup(name='ModelReport',
      version=version,
      description='A class to create pdf reports for classification algorithems',
      author='Tobias Rothlin',
      author_email='tobias@rothlin.com',
      url='https://github.com/BA202/ModelReport/ModelReport',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'matplotlib',
          'pdfkit',
          'seaborn',
      ],
     )
