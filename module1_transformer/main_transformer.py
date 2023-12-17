from sklearn.model_selection import train_test_split
from transformer import TransformerEncoder
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
import numpy as np


def main():

    # check out gpu
    def check_gpu():
        try:
            # Try to create a small random tensor on the GPU
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0, 3.0]])
            print("GPU is available.")
        except RuntimeError as e:
            print("GPU is not available: ", e)

    check_gpu()

    # importing data
    seqs_out_arr = np.load('./three_classes_combined_tokenized/feature_matrix_tokenized.npy')
    target_variable_arr = np.load('./three_classes_combined_tokenized/target_variable.npy')

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(seqs_out_arr, target_variable_arr, test_size=0.2, random_state=42, stratify=target_variable_arr)

    vocab_size = len(np.unique(seqs_out_arr)) # number of unique k-mers
    emb_sz = 32 # tune
    window_size = seqs_out_arr.shape[1] # number of k-mers in a sequence 

    model = TransformerEncoder(vocab_size=vocab_size, emb_sz=emb_sz, window_size=window_size,num_of_blocks=1)

    # compile
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-03)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # early stopping
    early_stopper = EarlyStopping(
        monitor='val_loss',
        min_delta=0.0001, # 0 is the default 
        patience=500,
        mode='min',
        restore_best_weights=True  
    )

    # class weights
    # class_weights = compute_class_weight(
    #     'balanced',
    #     classes=np.unique(y_train), 
    #     y=y_train
    # )

    # create a dict
    # class_weight_dict = {0: class_weights[0], 1: class_weights[1], 2: class_weights[2]}

    # train with early stopping and class weights
    model.fit(X_train, y_train, epochs=1000, batch_size=256, validation_split=0.2, 
            callbacks=[early_stopper])#, class_weight=class_weight_dict)

    # test
    total_loss, total_accuracy = model.evaluate(X_test, y_test)
    print(f"Concatenated data Test Loss: {total_loss}, Test Accuracy: {total_accuracy}")

    # Saving model
    model.save("./transformer_1block")

if __name__ == '__main__':
    main()
