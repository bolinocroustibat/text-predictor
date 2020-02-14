# Text Predictor

This is fork from of a character-level **RNN** (Recurrent Neural Net) **LSTM** (Long Short-Term Memory) implemented in Python 2.7/TensorFlow in order to predict a text based on a given dataset.

It's been updated to Python 3.6 / 3.7 compatibility and previous datasets were deleted.

Check out the original work here: https://github.com/gsurma/text_predictor

And see their corresponding Medium article here:

[Text Predictor - Generating Rap Lyrics with Recurrent Neural Networks (LSTMs)📄](https://towardsdatascience.com/text-predictor-generating-rap-lyrics-with-recurrent-neural-networks-lstms-c3a1acbbda79)


## To launch


For stupeflip lyrics generator:
```
python text_predictor.py stupeflip
```

For trolls generator:
```
python text_predictor.py comments
```

## In case of issues

In case of "RuntimeError: dictionary changed size during iteration":
https://github.com/tensorflow/tensorflow/issues/33183

Replace following line in linecache.py of your venv Python :
```
for mod in sys.modules.values():
```
with:
```
v = sys.modules.copy()
for mod in v:
```
