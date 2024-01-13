from pathlib import Path
import shutil

def initialize_project():
    curr_dir = Path.cwd().absolute()
    dist_folder = curr_dir / "dist"

    if dist_folder.exists():
        shutil.rmtree(f"{dist_folder}")
    else:
        dist_folder.mkdir(exist_ok=True)
    shutil.copytree('templates', 'dist/templates')
    files_for_dist = ['requirements.txt', 'README.md', 'pipeline_processor.py', 'detection_model.py', 'comments_preprocessor.py', 'app.py']
    for file in files_for_dist:
        shutil.copy(file, f"{dist_folder}")
    dataset_folder = dist_folder / "datasets"
    dataset_folder.mkdir(exist_ok=True)
    dataset_raw_folder = dataset_folder / "raw"
    dataset_raw_folder.mkdir(exist_ok=True)
    dataset_json_folder = dataset_folder / "json"
    dataset_json_folder.mkdir(exist_ok=True)
    static_folder = dist_folder / "static"
    static_folder.mkdir(exist_ok=True)
    static_plots_folder = static_folder / "plots"
    static_plots_folder.mkdir(exist_ok=True)
    shutil.copy('datasets/raw/youtube+spam+collection.zip', f"{dataset_raw_folder}")

    with open("./dist/model.bat", "w") as script:
        script.writelines([
            "@echo off",
            "set command=%1",
            "python -m venv .venv",
            "call .\.venv\Scripts\\activate.bat",
            "pip install -r requirements.txt",
            "python detection-model.py init"
        ])

    shutil.make_archive('dist', 'zip', 'dist/')

if __name__ == "__main__":
    initialize_project()