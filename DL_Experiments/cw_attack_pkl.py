from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.initializers import glorot_uniform
from keras.callbacks import ModelCheckpoint, EarlyStopping
import random
from keras.utils import np_utils
from keras.optimizers import Adamax
import numpy as np
import sys
import os
from timeit import default_timer as timer
from pprint import pprint
import argparse
import json
from data_utils_pkl import load_pkl_data

random.seed(0)


# define the ConvNet
class ConvNet:
    @staticmethod
    def build(classes,
              input_shape,
              activation_function=("elu", "relu", "relu", "relu", "relu", "relu"),
              dropout=(0.1, 0.1, 0.1, 0.1, 0.5, 0.7),
              filter_num=(32, 64, 128, 256),
              kernel_size=8,
              conv_stride_size=1,
              pool_stride_size=4,
              pool_size=8,
              fc_layer_size=(512, 512)):

        # confirm that parameter vectors are acceptable lengths
        assert len(filter_num) + len(fc_layer_size) <= len(activation_function)
        assert len(filter_num) + len(fc_layer_size) <= len(dropout)

        # Sequential Keras model template
        model = Sequential()

        # add convolutional layer blocks
        for block_no in range(0, len(filter_num)):
            if block_no == 0:
                model.add(Conv1D(filters=filter_num[block_no],
                                 kernel_size=kernel_size,
                                 input_shape=input_shape,
                                 strides=conv_stride_size,
                                 padding='same',
                                 name='block{}_conv1'.format(block_no)))
            else:
                model.add(Conv1D(filters=filter_num[block_no],
                                 kernel_size=kernel_size,
                                 strides=conv_stride_size,
                                 padding='same',
                                 name='block{}_conv1'.format(block_no)))

            model.add(BatchNormalization())

            model.add(Activation(activation_function[block_no], name='block{}_act1'.format(block_no)))

            model.add(Conv1D(filters=filter_num[block_no],
                             kernel_size=kernel_size,
                             strides=conv_stride_size,
                             padding='same',
                             name='block{}_conv2'.format(block_no)))

            model.add(BatchNormalization())

            model.add(Activation(activation_function[block_no], name='block{}_act2'.format(block_no)))

            model.add(MaxPooling1D(pool_size=pool_size,
                                   strides=pool_stride_size,
                                   padding='same',
                                   name='block{}_pool'.format(block_no)))

            model.add(Dropout(dropout[block_no], name='block{}_dropout'.format(block_no)))

        # flatten output before fc layers
        model.add(Flatten(name='flatten'))

        # add fully-connected layers
        for layer_no in range(0, len(fc_layer_size)):
            model.add(Dense(fc_layer_size[layer_no],
                            kernel_initializer=glorot_uniform(seed=0),
                            name='fc{}'.format(layer_no)))

            model.add(BatchNormalization())
            model.add(Activation(activation_function[len(filter_num)+layer_no],
                                 name='fc{}_act'.format(layer_no)))

            model.add(Dropout(dropout[len(filter_num)+layer_no],
                              name='fc{}_drop'.format(layer_no)))

        # add final classification layer
        model.add(Dense(classes, kernel_initializer=glorot_uniform(seed=0), name='fc_final'))
        model.add(Activation('softmax', name="softmax"))

        # compile model with Adamax optimizer
        optimizer = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(loss="categorical_crossentropy",
                      optimizer=optimizer,
                      metrics=["accuracy"])
        return model

def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Train and test the DeepFingerprinting model with PKL data in Closed-World setting.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data',
                        type=str,
                        required=True,
                        metavar='<path/to/ts_mon.pkl>',
                        help='Path to the ts_mon.pkl file.')
    parser.add_argument('-a', '--attack',
                        type=int,
                        default=0,
                        metavar='<attack_type>',
                        help='Type of attack: (0) direction (1) directional_timing (2) timing')
    parser.add_argument('-o', '--output',
                        type=str,
                        default='trained_model_cw_pkl.h5',
                        metavar='<output>',
                        help='Location to store the trained model.')
    return parser.parse_args()

def main():
    """
    Main function for PKL-based closed-world experiment
    """
    
    # Parse arguments
    args = parse_arguments()
    
    # Load PKL dataset
    print("Loading PKL dataset as type {}...".format(args.attack))
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_pkl_data(
        args.data, typ=args.attack
    )
    
    print(f"Train samples: {len(X_train)}")
    print(f"Validation samples: {len(X_valid)}")
    print(f"Test samples: {len(X_test)}")
    
    # Convert class vectors to categorical
    classes = len(set(list(y_train)))
    print(f"Number of classes: {classes}")
    
    y_train = np_utils.to_categorical(y_train, classes)
    y_valid = np_utils.to_categorical(y_valid, classes)
    y_test = np_utils.to_categorical(y_test, classes)
    
    # Build and compile model
    print("Compiling model...")
    model = ConvNet.build(classes=classes, input_shape=(5000, 1))
    
    # Train the model
    filepath = args.output
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, mode='auto', restore_best_weights=True)
    callbacks_list = [checkpoint, early_stopping]
    
    print("Training model...")
    history = model.fit(X_train, y_train,
                        epochs=50,
                        batch_size=32,
                        verbose=1,
                        validation_data=(X_valid, y_valid),
                        callbacks=callbacks_list)
    
    # Save model
    model.save(filepath)
    print(f"Model saved to {filepath}")
    
    # Evaluate model
    print("Evaluating model...")
    train_score = model.evaluate(X_train, y_train, verbose=0)
    test_score = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\n=> Train accuracy: {train_score[1]:.4f}")
    print(f"=> Test accuracy: {test_score[1]:.4f}")
    
    return test_score[1]

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
