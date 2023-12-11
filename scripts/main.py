import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
tf.get_logger().setLevel(100) # suppress deprecation messages
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical


import numpy as np
import dice_ml

import pandas as pd

import argparse

from util import CounterfactualOurs, Explanation, Stats, apply_noise, Dataset

np.random.seed(42)
tf.random.set_seed(42)


repeat_times = 3

def nn_model(input_shape):

    x_in = Input(shape=(input_shape,))
    x = Dense(20, activation='relu')(x_in)
    x = Dense(10, activation='relu')(x)
    x_out = Dense(2, activation='softmax')(x)
    nn = Model(inputs=x_in, outputs=x_out)
    nn.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return nn

def train_model(x_train, y_train):

    nn = nn_model(x_train.shape[1])
    nn.summary()
    nn.fit(x_train, y_train, batch_size=8, epochs=100, verbose=1)

    return nn

def main(args):

    # Loading data
    ds_name = args.dataset_name
    ds = Dataset(args.data_path, ds_name, args.cf_algo)

    if ds_name == "german":
        if args.cf_algo == "dice":
            x_train, y_train, x_test, y_test, d, feature_names = ds.load_german() 
        else:
            tf.compat.v1.disable_v2_behavior() # disable TF2 behaviour as alibi code still relies on TF1 constructs
            x_train, y_train, x_test, y_test = ds.load_german()
    else:

        if args.cf_algo == "dice":
            x_train, y_train, x_test, y_test, d, feature_names = ds.load_data() 
        else:
            tf.compat.v1.disable_v2_behavior() # disable TF2 behaviour as alibi code still relies on TF1 constructs
            x_train, y_train, x_test, y_test = ds.load_data()

    # Train and evaluate model
    if args.train:
        model = train_model(x_train, y_train)

        model.save(f'{args.model_path}nn_{ds_name}.h5', save_format='h5')
        tf.saved_model.save(model, f'{args.model_path}nn_{ds_name}_saved_model.h5') # used for onnx conversion
    else:
        model = tf.keras.models.load_model(f'{args.model_path}nn_{ds_name}.h5')

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy: ', score[1])

    # Instantiate stats object
    # Compute some stats on dataset
    # lower and upper bounds of features
    stats = Stats(x_train, model, args.total_cfx)
    norm = args.norm

    # Generate counterfactuals
    start = 50 # start 0 for validation, 50 for indipendent test
    X = x_test[start:start+args.nexps].reshape((args.nexps,) + x_test[1].shape)
    shape = X.shape

    # Counter used to select exactly n_exp number of explanations
    i = 0
    while i < shape[0]:
        for r in range(repeat_times):
            # Process one CFX at a time
            X_i = X[i]
            # Create a noisy copy of input
            X_i_noisy = apply_noise(X_i, stats)
            # Reshaping to so that libraries won't complain
            X_i = X_i.reshape(1,shape[1])
            X_i_noisy = X_i_noisy.reshape(1,shape[1])

            # Check that both X_i and X_i_noisy have the same label
            # if not, move on to next data instance
            
            l_i = np.argmax(model.predict(X_i.reshape(1, -1)), axis=1)[0]
            l_i_noisy = np.argmax(model.predict(X_i_noisy.reshape(1, -1)), axis=1)[0]

            if not(l_i == l_i_noisy):
                continue
            else:
                #set key to store result for current explanation
                stats.set_key(f"{i}{r}")


            if args.cf_algo == "dice":
                # First convert input back into dataframe    
        
                X_df = pd.DataFrame(X_i, columns=feature_names)
                
                # DICE ML wrapper for model
                m = dice_ml.Model(model=model, backend="TF2")

                stats.record_time(True)

                # Using method=random for generating CFs
                cf = dice_ml.dice.Dice(d, m, method="gradient")
                        
                # generate counterfactual

                # Set of diverse counterfactuals for factual input
                diverse_cfx = cf.generate_counterfactuals(X_df, total_CFs=args.total_cfx, desired_class="opposite").cf_examples_list[0]

                stats.record_time(False)

                # drop target column
                diverse_cfx = diverse_cfx.final_cfs_df.iloc[:, :-1].values.tolist()
                # Wrapping explanations into our Explanation class
                explanation = []

                for item in diverse_cfx:
                    e = Explanation()
                    e.cf['class'] = None
                    e_as_array = np.asarray(item)
                    e.cf['X'] = np.expand_dims(e_as_array, axis=0)
                    explanation.append(e)

                # Repeat for noisy input   
                X_df_noisy = pd.DataFrame(X_i_noisy, columns=feature_names)

                # Set of diverse counterfactuals for noisy input
                diverse_cfx_noisy = cf.generate_counterfactuals(X_df_noisy, total_CFs=args.total_cfx, desired_class="opposite").cf_examples_list[0]
                diverse_cfx_noisy = diverse_cfx_noisy.final_cfs_df.iloc[:, :-1].values.tolist()

                # Wrapping explanations into our Explanation class
                explanation_noisy = []

                for item in diverse_cfx_noisy:
                    e = Explanation()
                    e.cf['class'] = None
                    e_as_array = np.asarray(item)
                    e.cf['X'] = np.expand_dims(e_as_array, axis=0)
                    explanation_noisy.append(e)
        

            else: # ours

                # get model predictions
                predictions = np.argmax(model.predict(x_train), axis=1)

                alpha = args.alpha
                
                stats.record_time(True)

                cf = CounterfactualOurs(model, X_i.shape, args.total_cfx, norm, args.beta, args.gamma, args.opt)

                # fit kdtrees (one per class) based on model labels
                cf.fit_kdtree(x_train, predictions, 2)

                # Compute sets of diverse counterfactuals
                explanation = cf.explain(X_i, alpha) # used 1000 for others

                stats.record_time(False)

                explanation_noisy = cf.explain(X_i_noisy, alpha)  # used 1000 for others

            # Evaluate explanation
            stats.evaluate(X_i, X_i_noisy, explanation, explanation_noisy, norm)

        # Increment explanation counter
        i = i + 1
    
    ppresult = stats.get_stats_summary(args.cf_algo)

    with open(args.log_path, 'a') as f:
        f.write(ppresult)




if __name__ == "__main__":

    choices_algo = ["dice", "ours"]
    choices_data = ["german", "no2", "diabetes", "news", "spam"]


    parser = argparse.ArgumentParser(description='CFX generation script.')
    parser.add_argument('dataset_name', metavar='ds', default=None, help=f'Dataset name. Supported: {choices_data}', choices=choices_data)
    parser.add_argument('data_path', metavar='dp', default=None, help='Path to dataset.')
    parser.add_argument('model_path', metavar='mp', default=None, help='Path where model should be loaded/saved.')
    parser.add_argument('log_path', metavar='lp', default=None, help='Path where logs should be loaded/saved.')
    parser.add_argument('cf_algo', metavar='a', default=None, help=f'Explanation algorithms. Supported: {choices_algo}', choices=choices_algo)
    parser.add_argument('--train', action="store_true", help='Controls whether model is trained anew or loaded. Default: False.')
    parser.add_argument('--nexps', type=int, default=1, help='Number of data points to explain. Default: 1.')
    parser.add_argument('--total_cfx', type=int, default=5, help='Max number of diverse to be generated for each factual input. Default: 5.')
    parser.add_argument('--norm', type=int, default=1, help='Norm to be used to evaluate distances. Default: 1.')
    parser.add_argument('--alpha', type=int, default=50, help='Number of candidates to be selected. Default: 50.')
    parser.add_argument('--beta', type=float, default=0, help='Parameter used for diversity filter. Default: 0.')
    parser.add_argument('--gamma', type=float, default=0.01, help='Accuracy used for binary search. Default: 0.01.')
    parser.add_argument('--opt', action='store_true', help="Parameter used to control whether to minimise distance of CFXs or not. Default:False")



    args = parser.parse_args()

    main(args)