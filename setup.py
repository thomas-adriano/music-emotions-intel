from distutils.core import setup

setup(name='music-emotions-intel',
      version='0.1',
      description='A MIR tool',
      author='Thomas Adriano',
      author_email='thomas.o.adriano@gmail.com',
      url='https://github.com/thomas-adriano/music-emotions-intel.git',
      install_requires=[
          'librosa >= 0.5.0',
          'numpy >= 1.12.0',
          'scikit-learn >= 0.18.1',
          'pandas >= 0.19.2'
      ],
      )
