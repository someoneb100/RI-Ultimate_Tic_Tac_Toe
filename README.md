# RI-Ultimate_Tic_Tac_Toe

The goal of this project is to train an AI to play <a href="https://en.wikipedia.org/wiki/Ultimate_tic-tac-toe"> Ultimate Tic Tac Toe</a> with the <a href="https://en.wikipedia.org/wiki/Ultimate_tic-tac-toe#Rules">expanded rule-set</a> (playing a field on an already won mini-board is not allowed). The purpouse of adding this rule is to avoid making an agent that plays a high-risk high-reward strategy that always wins against unskilled players but newer defeats someone who knows how to play.

The model will be trained on Google Colaboratory (since i don't have a high performance GPU), but everything else will be done locally by simply downloading the stored weights of the already trained model. The training model involves adversarial DeepQ training (learning trough self-play) combined with Monte Carlo Tree Search.
