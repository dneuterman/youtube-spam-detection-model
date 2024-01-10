from argparse import ArgumentParser
from pathlib import Path
import subprocess
import shutil

def initialize_project():
    curr_dir = Path.cwd().absolute()
    print(curr_dir)

    dist_folder = curr_dir / "dist"
    dist_folder.mkdir(exist_ok=True)
    shutil.copytree('templates', 'dist/templates')
    files_for_dist = ['requirements.txt', 'README.md', 'pipeline_processor.py', 'detection_model.py', 'comments_preprocessor.py', 'app.py', 'setup.py']
    for file in files_for_dist:
        shutil.copy(file, f"{dist_folder}")
    dataset_folder = dist_folder / "datasets"
    dataset_folder.mkdir(exist_ok=True)
    dataset_folder = dataset_folder / "raw"
    dataset_folder.mkdir(exist_ok=True)
    shutil.copy('datasets/raw/youtube+spam+collection.zip', f"{dataset_folder}")

    shutil.make_archive('dist', 'zip', 'dist/')

if __name__ == "__main__":
    initialize_project()