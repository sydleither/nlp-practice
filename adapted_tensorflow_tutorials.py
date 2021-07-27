from preprocessing import train_test_val, standardization_tf

import io
import matplotlib.pyplot as plt
import tensorflow as tf
import tqdm

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras import Model
from tensorflow.keras.layers import Dot, Embedding, Flatten


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
    
    plot_model(history)
    save_word_embeddings(model, vectorize_layer)
    
    
#based on https://www.tensorflow.org/tutorials/text/word2vec
def word2vec_embeddings():
    text_ds = tf.data.TextLineDataset('../messages.txt').filter(lambda x: tf.cast(tf.strings.length(x), bool))
    vectorize_layer = TextVectorization(
        standardize=standardization_tf,
        max_tokens=4096,
        output_mode='int',
        output_sequence_length=10)
    vectorize_layer.adapt(text_ds.batch(1024))
    
    text_vector_ds = text_ds.batch(1024).prefetch(tf.data.AUTOTUNE).map(vectorize_layer).unbatch()
    sequences = list(text_vector_ds.as_numpy_iterator())
    targets, contexts, labels = generate_training_data(
        sequences=sequences,
        window_size=2,
        num_ns=4,
        vocab_size=4096,
        seed=42)
    dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
    dataset = dataset.shuffle(10000).batch(1024, drop_remainder=True)
    dataset = dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    word2vec = Word2Vec(4096, 128, 4)
    word2vec.compile(optimizer='adam',
                     loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                     metrics=tf.metrics.BinaryAccuracy(threshold=0.0))
    history = word2vec.fit(dataset, epochs=20)
    
    plot_model_no_val(history)
    save_word_embeddings(word2vec, vectorize_layer)
    

#from https://www.tensorflow.org/tutorials/text/word2vec#generate_training_data
def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
    # Elements of each training example are appended to these lists.
    targets, contexts, labels = [], [], []
    
    # Build the sampling table for vocab_size tokens.
    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)
    
    # Iterate over all sequences (sentences) in dataset.
    for sequence in tqdm.tqdm(sequences):
    
        # Generate positive skip-gram pairs for a sequence (sentence).
        positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
              sequence,
              vocabulary_size=vocab_size,
              sampling_table=sampling_table,
              window_size=window_size,
              negative_samples=0)
    
        # Iterate over each positive skip-gram pair to produce training examples
        # with positive context word and negative samples.
        for target_word, context_word in positive_skip_grams:
            context_class = tf.expand_dims(
                tf.constant([context_word], dtype="int64"), 1)
            negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                true_classes=context_class,
                num_true=1,
                num_sampled=num_ns,
                unique=True,
                range_max=vocab_size,
                seed=seed,
                name="negative_sampling")
    
            # Build context and label vectors (for one target word)
            negative_sampling_candidates = tf.expand_dims(
                negative_sampling_candidates, 1)
        
            context = tf.concat([context_class, negative_sampling_candidates], 0)
            label = tf.constant([1] + [0]*num_ns, dtype="int64")
        
            # Append each element from the training example to global lists.
            targets.append(target_word)
            contexts.append(context)
            labels.append(label)
    
    return targets, contexts, labels


#from https://www.tensorflow.org/tutorials/text/word2vec#model_and_training
class Word2Vec(Model):
    def __init__(self, vocab_size, embedding_dim, num_ns):
        super(Word2Vec, self).__init__()
        self.target_embedding = Embedding(vocab_size,
                                          embedding_dim,
                                          input_length=1,
                                          name="embedding")
        self.context_embedding = Embedding(vocab_size,
                                           embedding_dim,
                                           input_length=num_ns+1)
        self.dots = Dot(axes=(3, 2))
        self.flatten = Flatten()
    
    def call(self, pair):
        target, context = pair
        word_emb = self.target_embedding(target)
        context_emb = self.context_embedding(context)
        dots = self.dots([context_emb, word_emb])
        return self.flatten(dots)


#from https://www.tensorflow.org/tutorials/keras/text_classification#create_a_plot_of_accuracy_and_loss_over_time
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
    
    
def plot_model_no_val(history):
    history_dict = history.history
    
    acc = history_dict['binary_accuracy']
    loss = history_dict['loss']
    
    epochs = range(1, len(acc) + 1)
    
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.title('Training accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()
    
    
#from https://www.tensorflow.org/text/guide/word_embeddings
def save_word_embeddings(model, vectorize_layer):
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
    
    
#binary_classification_nn()
word2vec_embeddings()