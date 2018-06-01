import collections
import math
import os
import random
import zipfile
import numpy as np
from six.moves import urllib
from six.moves import xrange
# https://www.jianshu.com/p/bc2852978010

# pylint: disable=redefined-builtin
import tensorflow as tf
print(tf.__version__)
# Step 1: Download the data.
url = 'http://mattmahoney.net/dc/textdata'


def maybe_download(filename, expected_bytes):
    """ 
    Download a file if not present, 
    and make sure it's the right size. 
    """
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + filename +
            '. Can you get it with a browser')
    return filename


filename = maybe_download('text8.zip', 31344016)


def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


vocabulary = read_data(filename)
print('Data size', len(vocabulary))

vocabulary_size = 50000


def build_dataset(words, n_words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


data, count, dictionary, reverse_dictionary = build_dataset(
    vocabulary, vocabulary_size)
del vocabulary
print('Most common words (+UNK) ', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0


def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1
    # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window
        # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels


batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]], '->',
          labels[i, 0], reverse_dictionary[labels[i, 0]])

batch_size = 128
embedding_size = 400
skip_window = 10
num_skips = 2

valid_size = 16
valid_window = 100
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 10
graph = tf.Graph()
with graph.as_default():
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases, labels=train_labels,
                       inputs=embed, num_sampled=num_sampled,
                       num_classes=vocabulary_size))
# Construct the SGD optimizer using a learning rate of 1.0.
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, valid_dataset)
# Add variable initializer.
    similarity = tf.matmul(
        valid_embeddings, normalized_embeddings, transpose_b=True)
    init = tf.global_variables_initializer()
# Step 5: Begin training.
num_steps = 100001
with tf.Session(graph=graph) as session:
    # We must initialize all variables before we use them.
    init.run()
    print('Initialized')
    average_loss = 0
    for step in xrange(num_steps):
        batch_inputs, batch_labels = generate_batch(
            batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val
        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            print('Average loss at step ', step, ': ', average_loss)
            average_loss = 0

        if step % 10000 == 0:
            sim = similarity.eval()
            for i in xrange(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = 'Nearest to %s:' % valid_word
                for k in xrange(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = '%s %s,' % (log_str, close_word)
                print(log_str)
    final_embeddings = normalized_embeddings.eval()
'''
1.4.0
Found and verified text8.zip
Data size 17005207
Most common words (+UNK)  [['UNK', 418391], ('the', 1061396), ('of', 593677), ('and', 416629), ('one', 411764)]
Sample data [5236, 3084, 12, 6, 195, 2, 3134, 46, 59, 156] ['anarchism', 'originated', 'as', 'a', 'term', 'of', 'abuse', 'first', 'used', 'against']
3084 originated -> 12 as
3084 originated -> 5236 anarchism
12 as -> 3084 originated
12 as -> 6 a
6 a -> 195 term
6 a -> 12 as
195 term -> 6 a
195 term -> 2 of
2018-03-20 21:25:31.561180: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
Initialized
Average loss at step  0 :  57.7897949219
Nearest to as: trouser, dassault, sterility, tesla, synthesised, chupacabra, concise, supportive,
Nearest to d: slurs, girl, abbas, elaborating, pockets, troubled, woodcuts, warthog,
Nearest to has: phallic, syndicate, besson, pittsburgh, menagerie, stationary, praetors, lenten,
Nearest to some: princip, baggage, heraclea, culminates, lovable, kibbutz, interplay, articulated,
Nearest to after: flavors, unaware, prompt, vin, vanities, prosody, deuterocanonical, abbreviation,
Nearest to four: reloaded, modernists, insurrections, hosts, cnn, supergravity, nhl, antonia,
Nearest to that: whitish, oxus, fbi, eostre, condenses, yeah, yerba, lured,
Nearest to of: machina, laika, staples, baldwin, paton, between, pressburg, reformed,
Nearest to if: nematic, kazantzakis, hempstead, krypton, truk, davis, understandable, damascus,
Nearest to th: otimes, swampy, grossman, sumner, ecuadorian, cor, jl, surrogate,
Nearest to state: care, interacting, arrests, vn, tremblay, activision, tonsure, amanda,
Nearest to over: institute, fad, osmosis, rutherford, nobility, bysshe, chose, monegasque,
Nearest to about: cost, hanged, malaya, cosmetic, equine, other, tw, fv,
Nearest to eight: eminem, regret, tanaka, domitia, proportionately, invited, maitreya, observing,
Nearest to during: hipparchus, borer, vices, bathed, ordinary, neurotransmitters, factorizations, husbands,
Nearest to other: impression, irvine, monoamine, tyndall, about, tones, meta, venerate,
Average loss at step  2000 :  37.9820240941
Average loss at step  4000 :  28.5766845152
Average loss at step  6000 :  24.4274286268
Average loss at step  8000 :  21.4819719918
Average loss at step  10000 :  19.2604282647
Nearest to as: of, in, and, the, UNK, to, zero, a,
Nearest to d: girl, zero, agriculture, legislative, an, poor, alice, nine,
Nearest to has: the, in, and, of, zero, seven, to, one,
Nearest to some: a, to, and, the, one, UNK, in, nine,
Nearest to after: the, UNK, is, of, nine, released, during, a,
Nearest to four: nine, of, zero, a, one, in, and, UNK,
Nearest to that: a, the, and, UNK, to, of, in, one,
Nearest to of: the, and, in, UNK, a, to, nine, one,
Nearest to if: davis, the, of, to, in, black, red, further,
Nearest to th: remains, one, ayn, a, in, to, nine, the,
Nearest to state: care, a, to, converted, alkanes, and, november, agassi,
Nearest to over: the, institute, nine, ch, in, nobility, to, federal,
Nearest to about: other, of, cost, and, the, in, to, zero,
Nearest to eight: UNK, in, and, of, the, a, zero, nine,
Nearest to during: open, UNK, forms, after, amateur, river, technique, crater,
Nearest to other: of, and, nine, about, a, the, UNK, zero,
Average loss at step  12000 :  17.7611834471
Average loss at step  14000 :  16.4772721308
Average loss at step  16000 :  15.3564331145
Average loss at step  18000 :  14.1174710772
Average loss at step  20000 :  13.7509600483
Nearest to as: and, to, the, is, of, a, UNK, in,
Nearest to d: girl, nine, five, s, a, zero, legislative, cox,
Nearest to has: zero, seven, one, is, and, a, of, to,
Nearest to some: to, one, the, in, a, and, of, nine,
Nearest to after: is, the, two, three, zero, to, one, released,
Nearest to four: nine, one, zero, a, of, in, two, to,
Nearest to that: to, the, and, a, of, in, is, UNK,
Nearest to of: the, and, a, in, to, is, UNK, one,
Nearest to if: to, in, the, and, davis, of, as, zero,
Nearest to th: zero, the, nine, and, a, UNK, in, two,
Nearest to state: the, and, is, care, one, a, to, converted,
Nearest to over: zero, institute, the, two, chose, nobility, nine, one,
Nearest to about: two, zero, other, cost, is, and, the, of,
Nearest to eight: nine, one, in, the, UNK, zero, two, and,
Nearest to during: of, in, to, open, a, after, hipparchus, is,
Nearest to other: zero, to, a, of, UNK, and, two, the,
Average loss at step  22000 :  12.5550433906
Average loss at step  24000 :  12.1169283537
Average loss at step  26000 :  11.8695577658
Average loss at step  28000 :  11.3028374913
Average loss at step  30000 :  10.7501294265
Nearest to as: is, and, to, UNK, a, in, of, the,
Nearest to d: one, and, zero, six, s, to, two, four,
Nearest to has: of, seven, in, six, is, UNK, and, four,
Nearest to some: to, UNK, the, in, one, it, four, seven,
Nearest to after: to, two, of, for, in, five, and, as,
Nearest to four: one, two, nine, zero, six, seven, in, eight,
Nearest to that: to, a, and, in, of, is, the, by,
Nearest to of: the, is, in, to, and, a, UNK, by,
Nearest to if: to, in, and, that, one, was, as, davis,
Nearest to th: the, in, two, was, to, one, seven, and,
Nearest to state: UNK, and, a, is, one, care, to, of,
Nearest to over: and, of, two, institute, zero, six, american, for,
Nearest to about: zero, of, other, is, six, cost, three, for,
Nearest to eight: nine, one, in, six, two, three, zero, seven,
Nearest to during: the, in, of, to, was, and, one, by,
Nearest to other: of, a, to, UNK, the, and, is, three,
Average loss at step  32000 :  10.2438453663
Average loss at step  34000 :  9.79898616803
Average loss at step  36000 :  9.5368055687
Average loss at step  38000 :  9.051832569
Average loss at step  40000 :  8.71209394008
Nearest to as: in, and, a, is, to, by, of, for,
Nearest to d: four, three, the, two, in, zero, s, six,
Nearest to has: of, is, six, zero, seven, for, in, and,
Nearest to some: it, four, and, a, s, one, to, three,
Nearest to after: two, one, three, five, four, for, of, was,
Nearest to four: two, six, one, three, five, eight, seven, zero,
Nearest to that: to, a, in, the, and, by, of, one,
Nearest to of: the, and, in, a, s, one, is, by,
Nearest to if: as, to, in, for, of, that, UNK, and,
Nearest to th: two, four, UNK, in, one, seven, nine, eight,
Nearest to state: and, in, a, UNK, care, to, of, one,
Nearest to over: two, and, six, eight, UNK, four, nine, three,
Nearest to about: four, three, one, zero, two, five, to, other,
Nearest to eight: nine, one, two, seven, five, in, zero, six,
Nearest to during: in, zero, one, was, it, by, eight, with,
Nearest to other: of, four, in, is, and, one, three, a,
Average loss at step  42000 :  8.42085523474
Average loss at step  44000 :  8.3585880661
Average loss at step  46000 :  8.0018631705
Average loss at step  48000 :  8.0055964458
Average loss at step  50000 :  7.5431803208
Nearest to as: in, for, to, by, of, and, the, UNK,
Nearest to d: four, seven, in, and, six, the, s, of,
Nearest to has: is, in, as, seven, six, four, from, that,
Nearest to some: for, that, it, is, and, are, by, of,
Nearest to after: zero, for, three, was, as, five, s, four,
Nearest to four: six, five, two, eight, nine, one, zero, three,
Nearest to that: in, to, by, for, and, a, as, the,
Nearest to of: the, and, in, as, for, a, by, is,
Nearest to if: for, to, with, as, that, and, is, it,
Nearest to th: four, six, in, seven, nine, eight, one, two,
Nearest to state: zero, in, as, to, the, care, and, for,
Nearest to over: in, four, three, of, six, two, and, s,
Nearest to about: three, to, with, two, zero, as, in, four,
Nearest to eight: one, seven, zero, six, nine, two, five, three,
Nearest to during: in, five, and, was, the, seven, zero, it,
Nearest to other: of, in, the, four, a, with, to, and,
Average loss at step  52000 :  7.41208914614
Average loss at step  54000 :  7.25618191683
Average loss at step  56000 :  7.10520125818
Average loss at step  58000 :  6.84752841651
Average loss at step  60000 :  6.76424876572
Nearest to as: and, for, s, by, to, of, in, with,
Nearest to d: s, six, nine, seven, two, a, to, eight,
Nearest to has: of, from, two, as, is, in, and, with,
Nearest to some: and, in, for, are, that, one, to, s,
Nearest to after: s, for, and, as, one, zero, by, was,
Nearest to four: five, eight, one, six, nine, three, two, seven,
Nearest to that: to, a, for, of, by, and, in, is,
Nearest to of: the, and, in, with, is, a, by, that,
Nearest to if: to, that, one, with, it, for, as, from,
Nearest to th: nine, six, eight, one, by, four, zero, five,
Nearest to state: and, for, with, is, from, to, one, by,
Nearest to over: six, and, two, one, three, four, of, in,
Nearest to about: to, and, in, UNK, three, with, by, six,
Nearest to eight: six, one, seven, nine, five, four, three, two,
Nearest to during: four, on, eight, for, it, five, to, s,
Nearest to other: and, is, to, of, an, UNK, two, s,
Average loss at step  62000 :  6.56247356141
Average loss at step  64000 :  6.29373355114
Average loss at step  66000 :  6.05615640819
Average loss at step  68000 :  6.08985479486
Average loss at step  70000 :  6.10236226439
Nearest to as: and, in, the, to, is, of, for, by,
Nearest to d: a, s, of, and, six, seven, two, one,
Nearest to has: is, with, from, as, are, a, to, in,
Nearest to some: and, to, that, in, are, s, a, it,
Nearest to after: s, he, nine, with, for, by, was, and,
Nearest to four: six, eight, one, three, two, five, seven, nine,
Nearest to that: to, in, the, it, a, and, as, by,
Nearest to of: the, and, in, to, a, as, s, is,
Nearest to if: that, the, to, it, or, for, of, s,
Nearest to th: nine, zero, four, in, six, eight, seven, two,
Nearest to state: zero, with, for, and, the, from, at, is,
Nearest to over: to, s, a, six, of, in, with, american,
Nearest to about: one, zero, to, with, or, other, two, have,
Nearest to eight: nine, six, one, seven, four, three, five, two,
Nearest to during: and, to, nine, the, on, s, which, of,
Nearest to other: to, of, and, an, the, two, as, in,
Average loss at step  72000 :  5.86085396522
Average loss at step  74000 :  5.904252689
Average loss at step  76000 :  5.83429327989
Average loss at step  78000 :  5.76871449959
Average loss at step  80000 :  5.65152928799
Nearest to as: and, for, in, a, by, that, it, s,
Nearest to d: s, in, and, six, a, one, eight, seven,
Nearest to has: with, in, and, from, was, as, for, the,
Nearest to some: from, in, are, to, it, and, that, is,
Nearest to after: was, the, he, s, by, in, for, one,
Nearest to four: six, seven, three, two, five, one, eight, nine,
Nearest to that: to, this, for, it, and, of, which, a,
Nearest to of: in, and, the, for, that, with, an, to,
Nearest to if: that, not, it, for, or, two, be, s,
Nearest to th: eight, four, seven, nine, one, six, two, in,
Nearest to state: and, of, to, with, in, s, from, zero,
Nearest to over: zero, a, s, three, in, six, four, two,
Nearest to about: two, six, or, in, with, to, have, the,
Nearest to eight: nine, seven, six, one, four, five, three, two,
Nearest to during: was, of, the, one, five, it, three, this,
Nearest to other: to, as, or, and, are, in, on, for,
Average loss at step  82000 :  5.52909982276
Average loss at step  84000 :  5.58254582405
Average loss at step  86000 :  5.53356486702
Average loss at step  88000 :  5.39891414785
Average loss at step  90000 :  5.39648197603
Nearest to as: and, of, in, which, for, a, it, with,
Nearest to d: nine, s, two, one, UNK, six, eight, five,
Nearest to has: to, of, from, and, with, for, was, is,
Nearest to some: from, and, that, it, with, for, as, to,
Nearest to after: the, s, was, he, in, on, one, nine,
Nearest to four: three, six, eight, seven, nine, one, five, two,
Nearest to that: this, it, and, is, which, with, to, for,
Nearest to of: the, in, and, from, by, one, with, as,
Nearest to if: or, to, that, not, it, from, and, on,
Nearest to th: eight, four, six, one, seven, two, in, by,
Nearest to state: with, in, the, of, and, an, eight, s,
Nearest to over: three, in, zero, six, that, an, s, four,
Nearest to about: two, with, five, six, four, zero, from, have,
Nearest to eight: six, one, four, nine, three, seven, two, five,
Nearest to during: was, in, with, five, three, on, it, of,
Nearest to other: as, to, or, and, with, are, it, for,
Average loss at step  92000 :  5.1555163939
Average loss at step  94000 :  5.12198320097
Average loss at step  96000 :  5.0741420244
Average loss at step  98000 :  5.07910637122
Average loss at step  100000 :  5.06172936308
Nearest to as: a, to, of, by, that, which, or, this,
Nearest to d: a, two, and, six, seven, nine, one, UNK,
Nearest to has: is, of, a, in, to, for, from, by,
Nearest to some: to, and, are, that, from, as, for, it,
Nearest to after: in, was, s, and, to, with, two, by,
Nearest to four: five, two, six, one, three, seven, eight, nine,
Nearest to that: this, to, and, a, it, which, in, not,
Nearest to of: the, in, and, a, by, to, is, with,
Nearest to if: to, or, not, that, with, it, be, in,
Nearest to th: seven, by, eight, one, in, the, six, nine,
Nearest to state: in, three, to, a, it, and, with, was,
Nearest to over: of, s, one, zero, six, with, nine, in,
Nearest to about: in, a, with, zero, have, or, five, for,
Nearest to eight: one, nine, seven, six, four, three, five, two,
Nearest to during: with, in, an, which, was, four, three, a,
Nearest to other: are, or, of, as, to, for, and, that,
[Finished in 515.6s]
'''