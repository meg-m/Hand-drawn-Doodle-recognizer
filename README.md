# Hand-drawn-Doodle-recognizer

## Overview
This project is a web application which basically consosts of 4 tabs for-
1)  Digit Recognition- Uses MNIST dataset
2)  Alphabet Recognition- Uses matlab format of EMNIST dataset. 
    Available here: https://www.nist.gov/itl/iad/image-group/emnist-dataset
3)  Doodle Recognition:
    It is basically inspired from Quick, Draw! app: https://quickdraw.withgoogle.com/
    Dataset is open-sourced and different formats can be found here: https://github.com/googlecreativelab/quickdraw-dataset
    Numpy bitmap files(.npy) are used for this project:
    https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap
4)  Guess the sketch-
    Fun feature to play around with the quick draw dataset. Random image from the dataset will be generated and you need to guess it what it is.



## Dependencies

```sudo pip install -r requirements.txt```

## Usage

Once dependencies are installed, just run this to see it in your browser. 

To train respective models-
```python train_digit.py```

```python train_alphabet.py```

```python train_image.py``` 

```python train_guess.py```

To run the app: 
```python app.py```


## Credits
https://github.com/llSourcell/how_to_deploy_a_keras_model_to_production 

https://github.com/googlecreativelab/quickdraw-dataset
