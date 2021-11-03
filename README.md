# NLP
Natural Language Processing project

## fasttext.py
This file is the FastText part of our assigment. We had to run it on our own pc as the colab didn't have enough RAM to run it in there. The program will open database.json file and calculates the Cosine similarities using vectors created using the FastText-model. After that it will output cosines.json which can be copied into the database back in the colab

How to run:
```
python fasttext.py
```

## GUI.py
This is our GUI for showing the results. It opens the database.json file and creates Qt window with tabs for each resulting plots

How to run:
```
python GUI.py
```