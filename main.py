import numpy as np
import psycopg2
import tensorflow.compat.v1 as tf
from psycopg2 import OperationalError
import datetime

tf.disable_v2_behavior()

# DB


def create_connection(db_name, db_user, db_password, db_host, db_port):
    conn = None
    try:
        conn = psycopg2.connect(
            database=db_name,
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port,
        )
        print("Connection to PostgreSQL DB successful")
    except OperationalError as e:
        print(f"The error '{e}' occurred")
    return conn


def execute_read_query(conn, query):
    cursor = conn.cursor()
    result = None
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        return result
    except OperationalError as e:
        print(f"The error '{e}' occurred")


connection = create_connection("smartrecipes", "postgres", "postgres", "localhost", "5432")

# CORPUS (INPUT)

# corpus_raw = 'He is the king . The king is royal . She is the royal  queen '.lower()

foodstuff_id_results = execute_read_query(connection, "SELECT id FROM dbo.foodstuff")
words = list(map(lambda r: r[0], foodstuff_id_results))

# CREATE WORD-INT MORPHISM

# words = []
# for word in corpus_raw.split():
#     if word != '.':
#         words.append(word)
#
# words = set(words)

word2int = {}
int2word = {}

vocab_size = len(words)
for i, word in enumerate(words):
    word2int[word] = i
    int2word[i] = word

# CREATE SENTENCES (aka DOCUMENTS)

# raw_sentences = corpus_raw.split('.')
# sentences = []
# for sentence in raw_sentences:
#     sentences.append(sentence.split())

# GENERATE TRAINING DATA
# I can probably get this data very easily I just take all tuples from all recipes of recipe ids

ingredient_tuple_results = execute_read_query(connection, """
    SELECT a.foodstuffid, b.foodstuffid
    FROM dbo.ingredient a
    INNER JOIN dbo.ingredient b ON b.recipeid = a.recipeid AND b.foodstuffid <> a.foodstuffid AND a.foodstuffid > b.foodstuffid
    LIMIT 100000
""")
data = list(map(lambda r: [r[0], r[1]], ingredient_tuple_results))
# WINDOW_SIZE = 2
#
# for sentence in sentences:
#     for word_index, word in enumerate(sentence):
#         for nb_word in sentence[max(word_index - WINDOW_SIZE, 0) : min(word_index + WINDOW_SIZE, len(sentence)) + 1] :
#             if nb_word != word:
#                 data.append([word, nb_word])

# TRANSFORM DATA TO ONE-HOT ENCODED TUPLES


def to_one_hot(data_point_index, vocabulary_size):
    temp = np.zeros(vocabulary_size)
    temp[data_point_index] = 1
    return temp


x_train = []  # input word
y_train = []  # output word

for data_word in data:
    x_train.append(to_one_hot(word2int[data_word[0]], vocab_size))
    y_train.append(to_one_hot(word2int[data_word[1]], vocab_size))

# convert them to numpy arrays
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

# CREATE THE MODEL

x = tf.placeholder(tf.float32, shape=(None, vocab_size))
y_label = tf.placeholder(tf.float32, shape=(None, vocab_size))

EMBEDDING_DIM = 32  # you can choose your own number

W1 = tf.Variable(tf.random_normal([vocab_size, EMBEDDING_DIM]))
b1 = tf.Variable(tf.random_normal([EMBEDDING_DIM]))  # bias
hidden_representation = tf.add(tf.matmul(x, W1), b1)

W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, vocab_size]))
b2 = tf.Variable(tf.random_normal([vocab_size]))
prediction = tf.nn.softmax(tf.add( tf.matmul(hidden_representation, W2), b2))

# TRAIN THE MODEL

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init) # make sure you do this!

# define the loss function:
cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), reduction_indices=[1]))

# define the training step:
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy_loss)
n_iters = 10000

# train for n_iter iterations
for _ in range(n_iters):
    sess.run(train_step, feed_dict={x: x_train, y_label: y_train})
    print('loss is : ', sess.run(cross_entropy_loss, feed_dict={x: x_train, y_label: y_train}))

# THE OUTPUT REPRESENTATION

# In our case we have also included a bias term b1 so you have to add it.
vectors = sess.run(W1 + b1)
# print(vectors)
# print(vectors[word2int['queen']])


def euclidean_dist(vec1, vec2):
    return np.sqrt(np.sum((vec1-vec2)**2))


def find_closest(word_index, vectors):
    min_dist = 10000  # to act like positive infinity
    min_index = -1
    query_vector = vectors[word_index]
    for index, vector in enumerate(vectors):
        if euclidean_dist(vector, query_vector) < min_dist and not np.array_equal(vector, query_vector):
            min_dist = euclidean_dist(vector, query_vector)
            min_index = index
    return min_index


for word in words:
    word_index = word2int[word]
    print(word + "," + str(vectors[word_index]) + "," + int2word[find_closest(word_index, vectors)])

print(datetime.datetime.now())
# VISUALIZE


