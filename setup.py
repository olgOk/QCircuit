from distutils.core import setup
setup(
  name = 'qcircuit',         # How you named your package folder
  packages = ['qcircuit'],   # Chose the same as "name"
  version = '1.0.1',      # Start with a small number and increase it with every change you make
  license='Apache',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Quantum Circuit Simulator',   # Give a short description about your library
  author = 'Olga Okrut',                   # Type in your name
  author_email = 'vokrut42sv@gmail.com.com',      # Type in your E-Mail
  url = 'https://github.com/olgOk/QCircuit',   # Provide either the link to your github or to your website
  python_requires = ('>=3.6.0'),
  download_url = "https://github.com/olgOk/QCircuit/archive/1.0.1.tar.gz"
)
