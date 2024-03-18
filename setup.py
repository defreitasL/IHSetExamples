from setuptools import setup, find_packages

setup(
    name='IHSetExamples',
    version='0.0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'matplotlib',
        'xarray',
        'spotpy',
        'IHSetCalibration @ git+https://github.com/defreitasL/IHSetCalibration.git',
        'IHSetJaramillo20 @ git+https://github.com/defreitasL/IHSetJaramillo20.git'
    ],
    author='Lucas de Freitas Pereira',
    author_email='lucas.defreitas@unican.es',
    description='IH-SET Jaramillo et al. (2020)',
    url='https://github.com/defreitasL/IHSetExamples',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)