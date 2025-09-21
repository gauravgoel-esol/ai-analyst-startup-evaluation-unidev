from setuptools import setup, find_packages
import os

# Helper to get long description from README, if present
def read_file(fname):
    here = os.path.abspath(os.path.dirname(__file__))
    try:
        with open(os.path.join(here, fname), encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""
def parse_requirements(filename):
    with open(filename, "r") as f:
        return f.read().splitlines()

setup(
    name="Ventures_Capital",
    version="0.1.0",
    author="ArunSinghNegi",
    author_email="negi.arun@otomashen.com",
    description="AI Startup Evaluation & Analyst Tools",
    long_description=read_file("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/gauravgoel-esol/ai-analyst-startup-evaluation-unidev",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=parse_requirements("requirements.txt"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Researchers",
        "Topic :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries",
        "Programming Language :: Python :: 3",
    ],
    include_package_data=True,
    zip_safe=False,
    entry_points={
    "console_scripts": [
        "vc_evaluate=app:main",  
        "vc_train=training:train_pipeline",  
        "vc_server=server:main",  
        "vc_preprocess=preprocessing:run_preprocessing",  
    ],
},
)
