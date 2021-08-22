from setuptools import setup, find_packages

with open('README.md') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='sota_music_taggers',
    version="0.0.1",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.7',
        'Topic :: Artistic Software',
        'Topic :: Multimedia',
        'Topic :: Multimedia :: Sound/Audio',
        'Topic :: Multimedia :: Sound/Audio :: Editors',
        'Topic :: Software Development :: Libraries',
    ],
    description='SOTA Music Taggers',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Minz Won',
    author_email='',
    maintainer='E. Manilow',
    maintainer_email='',
    url='https://github.com/ethman/sota-music-tagging-models',
    license='MIT',
    packages=find_packages(),
    package_data={'': ['models/*/*/best_model.pth', 'tag_datafiles/*']},
    keywords=['music tagging'],
    install_requires=requirements,
)

