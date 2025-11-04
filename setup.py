from setuptools import setup, find_packages

setup(
    name="ellama",
    version="0.1.0",
    packages=["convert", "data_process", "inference", "training"],
    package_dir={
        "": "src",                              # корневая директория
        "convert": "src/convert",              # или можно указать явно
        "data_process": "src/data_process",
        "inference": "src/inference",
        "training": "src/training"
    }
)