import numpy as np

from utility.mylogger import MyLogger

def countSamples(probability_vector, n_class):
    assert len(probability_vector.shape) == 2
    assert n_class >= 1

    table = np.zeros(n_class, dtype="int32")
    for item in probability_vector:
        table[np.argmax(item)] += 1

    logger = MyLogger.getLog()
    logger.debug("Statistics:")
    logger.debug("Total: %s", probability_vector.shape[0])
    for item in range(0, n_class):
        logger.debug("\tLabel %s: %s samples (%s percentage)", item, table[item], table[item] / len(
            table))

