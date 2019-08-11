import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='DataFrameLabeler',
    version='0.0.1',
    scripts=[],
    author='Martin Marenz',
    author_email='martin.marenz@gmail.com',
    description='An ipywidget helper class to manually label rows in pandas data frames.',
    license='MIT',
    long_description_content_type='text/markdown',
    long_description=long_description,
    url='https://github.com/Blonck/DataFrameLabeler',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
