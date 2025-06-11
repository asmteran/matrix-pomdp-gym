import setuptools

VERSION = "0.1.0"

short_description = (
    "An OpenAI gym / Gymnasium environment for discrete-action, "
    "continuous-observation Partially Observable Markov Decision Processes "
    "(POMDPs) using matrices."
)

REQUIRED_PACKAGES = [
    "gymnasium >= 0.26.2",
    "numpy >= 1.26.4",
]


# Loading the "long description" from the projects README file.
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = short_description  # Fallback to short description
    print("Warning: The file 'README.md' was not found. Falling back to using the short description as the long description.")

setuptools.setup(
    name="matrix-pomdp-gym",
    version=VERSION,
    author="A. Saleh Mteran",
    author_email="a.salehmteran@gmail.com",
    description=short_description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/asmteran/matrix-pomdp-gym",
    # Contained modules and scripts:
    packages=setuptools.find_packages(),
    install_requires=REQUIRED_PACKAGES,
    # PyPI package information:
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    license="MIT License",
    python_requires=">=3.6",
    keywords=' '.join([
        "Reinforcement-Learning",
        "Reinforcement-Learning-Environment",
        "Gym-Environment",
        "Markov-Decision-Processes",
        "Gym",
        "OpenAI-Gym",
        "Gymnasium",
    ]),
)