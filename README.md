# Youtube Spam Detection Model
A multinomial Naive Bayes classification model to detect YouTube comment spam. This is a project for school that mainly focuses on using sci-kit learn to generate a spam detection model to process YouTube comments. A flask frontend is created that allows the user to view statistics about the trained model and the data used to test it. It also provides the user the capability to upload their own comments and save the comments that are predicted as spam by the model. The saved comments can then be used to delete specific YouTube comments in a comment section.


## Installation and running the model
There are two ways to build and run the model. Either through the provided python files as a sort of development mode or using the provided **model.bat** file from the created **spam-detection-dist.zip** file.

> [!IMPORTANT]
> This application has only been tested for Windows 10 and 11 systems. Running it in a Mac or Linux environment may result in it not functioning properly. The provided installation steps assume that you will be installing this on a Windows machine. You should update your python installation to the latest version before running this program. It should work on python versions 3.12 and later.

### Running as development
1. After cloning the repository to your destination folder, you should first initialize a virtual environment to run the program and to not pollute your python packages on your local environment. Run the following commands in the command prompt terminal in your destination folder directory to create your environment:

    ```
    python -m venv .venv
    ```

    Activate it using the following command:

    ```
    .\.venv\Scrips\activate
    ```

    Your terminal should now be showing the activated virtual environment in parentheses next to you current working directory. For example:

    ```
    (.venv) C:\Users\...
    ```

    Remember to activate your virtual environment prior to running any other commands.
2. Now you can install the required packages using the provided **requirements.txt** file and running the following command:

    ```
    pip install -r requirements.txt
    ```

    Your terminal will show the progress of all the dowloads for each package. It may take a minute or two depending on your internet to fully complete.

3. After all the packages have been downloaded, you can now initialize the raw data in the **datasets** folder by extracting it from the provided .zip file. Run the command:

    ```
    python detection_model.py init
    ```

    This will extract the .zip file of the raw data to the **datasets/raw** folder in the repository.

4. Once the datasets have been extracted, you can now build the spam detection model by running:

    ```
    python detection_model.py build
    ```

5. Finally, once the model has finished being built, you can now start the flask server frontend by running:

    ```
    flask run
    ```
    The default url and port for the running server is [http://127.0.0.1:5000](http://127.0.0.1:5000). You can now access the application at this url. To close the sever, type **CTRL+C** in the terminal and follow the prompt to terminate the program.

### Creating a distribution file
1. After cloning the repository to your destination folder, run the following command in your terminal in the destination folder directory:

    ```
    python setup.py
    ```

    This will create a distribution file called **spam-detection-dist.zip** in your current working directory. This file can be shared and distributed for ease of building and running the detection model.

### Running a distribution file
1. Extract the **spam-detection-dist.zip** in whichever directory you have saved it.

2. Open a command prompt terminal in your current working directory and run the following command:

    ```
    model.bat build
    ```
    This will create your virtual environment, start it and then initialize and build the spam detection model.

3. To start the application, run the following command:

    ```
    model.bat run
    ```

    This will start the flask application that can be accessed at the url: [http://127.0.0.1:5000](http://127.0.0.1:5000). To close the sever, type **CTRL+C** in the terminal and follow the prompt to terminate the program.

## Using the application
When you first open the application, you are greeted with a welcome page that details an overview of the application as well as the functionality of each page. The prediction page allows you to upload a json file containing comments to be predicted. They should be in the same format as detailed on the welcome page. A set of sample comments in a json file is also available in the **datasets/json** folder to test comment uploads and predictions.
