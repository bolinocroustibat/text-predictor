# Text Predictor

https://github.com/thibo73800/tensorflow2.0-examples/blob/master/RNN%20-%20Text%20Generator.ipynb

some other sources :

https://towardsai.net/p/deep-learning/create-your-first-text-generator-with-lstm-in-few-minutes

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

