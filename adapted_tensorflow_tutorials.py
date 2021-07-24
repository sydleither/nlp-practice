from preprocessing import train_test_val

import io
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


#based on https://www.tensorflow.org/tutorials/keras/text_classification
def binary_classification_nn():
    X_train, y_train, X_test, y_test, X_val, y_val = train_test_val()
    
    vectorize_layer = TextVectorization(
        max_tokens=10000,
        output_mode='int',
        output_sequence_length=250)
    vectorize_layer.adapt(X_train)
    
    train_vector = vectorize_layer(X_train)
    test_vector = vectorize_layer(X_test)
    val_vector = vectorize_layer(X_val)

    model = tf.keras.Sequential([
        layers.Embedding(10000 + 1, 16, name='embedding'),
        layers.Dropout(0.2),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.2),
        layers.Dense(1)])

    model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
        optimizer='adam',
        metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

    history = model.fit(train_vector, y_train, validation_data=(val_vector, y_val), epochs=10)
    loss, accuracy = model.evaluate(test_vector, y_test)
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)
    
    plot_model(history)
    word_embeddings(model, vectorize_layer)
    
    
#https://www.tensorflow.org/tutorials/text/word2vec
def word2vec_embeddings():
    return
    
#taken from https://www.tensorflow.org/tutorials/keras/text_classification#create_a_plot_of_accuracy_and_loss_over_time
def plot_model(history):
    history_dict = history.history
    
    acc = history_dict['binary_accuracy']
    val_acc = history_dict['val_binary_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    
    epochs = range(1, len(acc) + 1)
    
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()
    
    
#taken from https://www.tensorflow.org/text/guide/word_embeddings
def word_embeddings(model, vectorize_layer):
    weights = model.get_layer('embedding').get_weights()[0]
    vocab = vectorize_layer.get_vocabulary()
    
    out_v = io.open('vectors.tsv', 'w', encoding='utf-8')
    out_m = io.open('metadata.tsv', 'w', encoding='utf-8')
    
    for index, word in enumerate(vocab):
      if index == 0:
        continue
      vec = weights[index]
      out_v.write('\t'.join([str(x) for x in vec]) + "\n")
      out_m.write(word + "\n")
    out_v.close()
    out_m.close()
    
    
binary_classification_nn()