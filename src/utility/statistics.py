import numpy as np

from utility.mylogger import MyLogger

def countSamples(probabilityVector, nClasses):
    assert len(probabilityVector.shape) == 2
    assert nClasses >= 1

    table = np.zeros(nClasses, dtype="int32")
    for item in probabilityVector:
        table[np.argmax(item)] += 1

    logger = MyLogger.getLog()
    logger.debug("Statistics:")
    for item in range(0, nClasses):
        logger.debug("\tLabel %s: %s samples (%s percentage)", item, table[item], table[item] / len(
            table))
