import setuptools


setuptools.setup(
    name='bsmu.bone-age',
    version='0.0.1',
    author='Ivan Kosik',
    author_email='ivankosik91@gmail.com',

    package_dir={'': 'src'},
    packages=setuptools.find_namespace_packages(where='src')
)
