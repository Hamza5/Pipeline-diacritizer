import setuptools

with open('README.md') as readme_file:
    long_desc = readme_file.read()

setuptools.setup(
    name='pipeline_diacritizer',
    version='1.0.2',
    author='Hamza Abbad',
    author_email='hamza.abbad@whut.edu.cn',
    description='Command-line application to automatically restore the diacritics of an Arabic text.',
    long_description=long_desc,
    long_description_content_type='text/markdown',
    url='https://github.com/Hamza5/Pipeline-diacritizer',
    packages=setuptools.find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Text Processing :: Linguistic'
    ],
    entry_points={'console_scripts': ['pipeline_diacritizer = pipeline_diacritizer.pipeline_diacritizer:main']},
    install_requires=['tensorflow<=1.14.0,>=1.11.0', 'numpy<=1.16.5,>=1.13.0'],
    python_requires='>=3.4,<3.8'
)
