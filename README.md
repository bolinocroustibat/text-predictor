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


## To run in the background on a distant server
Use tmux.

See https://www.howtogeek.com/671422/how-to-use-tmux-on-linux-and-why-its-better-than-screen/

Create a named session:
```
tmux new -s session-name
```

Detach from the session:
`Ctrl+b+d`

List sessions:
```
tmux ls
```

Attach to last session:
```
tmux a
```

Attach to a session:
```
tmux a -t session-name
```
