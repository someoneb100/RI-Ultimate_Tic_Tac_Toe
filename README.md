# RI-Ultimate_Tic_Tac_Toe

The goal of this project is to train an AI to play <a href="https://en.wikipedia.org/wiki/Ultimate_tic-tac-toe"> Ultimate Tic Tac Toe</a> with the <a href="https://en.wikipedia.org/wiki/Ultimate_tic-tac-toe#Rules">expanded rule-set</a> (playing a field on an already won mini-board is not allowed). The purpouse of adding this rule is to avoid making an agent that plays a high-risk high-reward strategy that always wins against unskilled players but newer defeats someone who knows how to play. 

The playing agent involves adversarial DeepQ training (learning trough self-play) combined with Monte Carlo Tree Search. The Neural Network (model) consists of CNN layers followed by Dense Layers. In `config.py` the training parameters can be set, and by running `coach.py` training is activated. The sqlite database `log.db` keeps track of stats. During training, a model that wins against the previous best model more time than it loses is pronounced the new best model. Loading of previous models is done by using `model_handler.py` functions. We chose not to upload all intermediate models to github, only the best one.

Because of the fact that the Monte Carlo Tree Search algorithm uses the model every time it tries another node in the tree, the bottle-neck of self-play turned out to be the playing itself, where for 600 step simulations per action playing a single game takes about 27 minutes. For playing 700 games of self-play it took us 2 weeks of non-stop training, which according to the AlphaZero research paper is not enough and it would take us a month to get any obvious results. We have set up a server in our living room attempting to let the model train without taking up one of our every-day machines. In an attempt to make the self-play parallel and across multiple devices, a client-server architecture was implemented but that made another even worse bottle-neck: the local network. If any device had multiple NVidia GPU-s the training would have perhaps been faster, but we don't have a device like that.

For testing purposes we made a model that uses pure MTCS without a neural network (predict just claims that all actions have equal probability of play, and only trough the tree search the probabilities change by finding winning moves). At the current state, a skilled player can defeat the model, but the model often wins against the testing model without a neural network (in our test out of 10 games he wins 6, ties 3 and loses 1, he plays equal times X and O). Testing is done by running `test.py`.

We also implemented a simple playing environment in Qt for demonstration purposes. After running `mainwindow.py` select one of the playing options and play against the model or watch two models play against each other.
