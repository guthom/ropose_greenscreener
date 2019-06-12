import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='ropose_greenscreener',
     version='0.1',
     scripts=[] ,
     author="Thomas Gulde",
     author_email="thomas.gulde@reutlingen-university.de",
     description="A backgorund augmentations tool",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/guthom/ropose_greenscreener",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3.6",
         "License :: OSI Approved :: MIT License",
     ],
 )
