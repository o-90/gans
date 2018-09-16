from setuptools import setup
from setuptools import find_packages


PKG_NAME = "gans"
VERSION = "0.0.1"
AUTHOR = "John Martinez"
EMAIL = "john.r.martinez14@gmail.com"
PKG_URL = "https://gitlab.com/gobrewers14/gans"
# LICENSE = ""

setup(name=PKG_NAME,
      version=VERSION,
      description="",
      long_description=open("README.md").read(),
      author=AUTHOR,
      author_email=EMAIL,
      url=PKG_URL,
      packages=find_packages(),
      zip_safe=False,
      setup_requires=["setuptools_scm"],
      install_requires=["numpy", "tqdm", "jupyter"])
