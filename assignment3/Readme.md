# Deep Learning Rnn

## Run Instruction For Part 3

## Train

`python bilstmTrain.py repr trainFile modelFile --dev_path devFile --task task --accuracy_logging_file_path accuracy_logging_file_path` where:

- `repr`: should be 'a','b','c' or 'd' based on the wanted represenation
- `trainFile`: is the path to train file (Please provide full path)
- `modelFile`: is a path to save the model's state after training (Please provide full path)
- `--dev_path`: is an optional parameter for a dev file path (Needed for live indication about the model's performence)
- `--task`: optional- gets only 'ner' / 'pos' - Important - if dev_path was provided please provide this as well to get a real performance indications
- `--accuracy_logging_file_path`: optional- log accuracy results to this file

## Predict

`python bilstmPredict.py repr modelFile inputFile --output_path output_path` where:

- `repr`: should be 'a','b','c' or 'd' based on the trained represenation
- `modelFile`: is a path to load the model's state after training (Please provide full path)
- `inputFile`: is a path to non labeled inputs to get predictions for them (Please provide full path)
- `output_path`: optional- it is an output path for the predicted tags (By default it write the tags to `./output.txt`)
