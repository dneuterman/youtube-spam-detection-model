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
            "@echo off\n",
            "set command=%1\n",
            r'IF "%command%"=="" (' + "\n",
            'ECHO Please provide "build" or "run" as an argument.\n',
            'ECHO Example: "model.bat build" will build the spam detection model.\n',
            'ECHO "model.bat run" will start the prrogram that can be accessed in your browser.\n',
            ")\n",
            r'IF "%command%"=="build" (' + "\n"
            "ECHO Creating the virtual environment and building the Multinomial Naive Bayes Spam Detection Model.\n",
            "python -m venv .venv\n",
            r"call .\.venv\Scripts\activate.bat" + "\n",
            "pip install -r requirements.txt\n",
            r".\.venv\Scripts\python.exe detection_model.py init" + "\n",
            r".\.venv\Scripts\python.exe detection_model.py build" + "\n",
            ")\n",
            r'IF "%command%"=="run" (' + "\n",
            r"call .\.venv\Scripts\activate.bat" + "\n",
            "ECHO Running the Spam Detection Model\n",
            "flask run\n",
            ")\n"
        ])

    shutil.make_archive('spam-detection-dist', 'zip', 'dist/')

if __name__ == "__main__":
    initialize_project()