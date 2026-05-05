"""Baselines for repetition counting.

Each baseline is a self-contained script that loads the same datasets as
``tasks/repetition/train.py`` and reports OBO accuracy + MAE on train and
test, so we can put a number on "how far above chance is the CTM run".

Available baselines:
  - modal_count: always predict the modal count from the train set.
                 No model, no decoding — chance floor for every metric.

Planned: rnn / lstm / cnn3d / transformer over pooled ResNet features.
"""
