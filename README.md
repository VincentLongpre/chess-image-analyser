# Chess Image Analyser

This project aims to simplify the process of analysing chess positions found in videos or books using analysis tools like the [lichess analysis board](https://lichess.org/analysis). The application takes an image of the position as input, extracts the information and redirects the user to a preloaded analysis board. This work builds on the 2018 [chessputzer software](https://github.com/metterklume/chessputzer) by improving reliability on images from web games that have become much more popular in the recent years.

## Installation and Launch

First, the environment must be installed using the following command:

```sh
$ pip install -r requirements.txt
```

Secondly, the dataset used to train the model can be found and easily downloaded on this [kaggle page](https://www.kaggle.com/datasets/koryakinp/chess-positions).

Once the data is downloaded, the model is trained using the procedure in [training.ipynb](training.ipynb). The model obtained is stored in the models folder and can then be saved to comet_ml. Script for comet_ml model saving can be found in [prediction.py](prediction.py)

Next, the assets used to generate images of the predicted position must be downloaded and saved in the directory (the path to the PNGs 1x with shadow images must be specified in the [chess_board.py](chess_board.py) file). The required assets are free to available on this [link](https://opengameart.org/content/chess-pieces-and-board-squares). 

Finally, we are ready to run the flask application that will serve the streamlit web application. To do so, run the following command:
```sh
$ gunicorn --bind 0.0.0.0:8080 app:app
```
Then, the streamlit app can be launched like this:
```sh
$ streamlit run streamlit_app.py
```

## Usage and Warnings

Once the application is running, cropped images of chess positions can be uploaded using the file system. The model performs best using images from online games using common piece sets. Additionally, if the image uploaded is not cropped to contain only the board, results might be unrepresentative of the input. 

After the model predictions are returned, the actual and predicted positions are displayed and a link to a preloaded lichess analysis board is created. Additional information about the position can be provided by the user (which color is playing next and what color is the user playing) using selection menus.

