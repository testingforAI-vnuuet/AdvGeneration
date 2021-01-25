from os import listdir

import numpy as np
import pandas as pd

from utility.mylogger import MyLogger

logger = MyLogger.getLog()

if __name__ == '__main__':
    ADV_STATISTICS_PATH = '/Users/ducanhnguyen/Documents/python/pycharm/AdvGeneration/data/mnist/matrix.csv'
    L2_PATH = '/Users/ducanhnguyen/Documents/python/pycharm/AdvGeneration/data/mnist/l2.csv'
    ADV_FOLDER = '/Users/ducanhnguyen/Documents/python/pycharm/AdvGeneration/data/mnist/images'

    n_adv_matrix = np.ndarray(shape=(10, 10), dtype=np.int)
    l2_matrix = np.ndarray(shape=(10, 10), dtype=np.float)
    for f in listdir(ADV_FOLDER):
        full_path = ADV_FOLDER + '/' + f
        source_label = int(f[:1])
        target_label = int(f[5:])
        if source_label != target_label:
            # compute the number of adv
            n_adv = int((len(listdir(full_path)) - 1) / 3)
            n_adv_matrix[source_label, target_label] = n_adv

            # compute l2 distance
            l2_arr = []
            summary_path = full_path + '/summary.csv'
            df = pd.read_csv(summary_path, delimiter=',').to_numpy()
            n_row = len(df)
            if n_row >= 1:
                L2_INDEX = 2
                for i in range(n_row):
                    l2_arr.append(df[i][L2_INDEX])
                l2_matrix[source_label, target_label] = np.average(l2_arr)

    np.savetxt(ADV_STATISTICS_PATH,
               n_adv_matrix.astype(int),
               fmt='%i',
               delimiter=',')
    np.savetxt(L2_PATH,
               l2_matrix,
               fmt='%1.2f',
               delimiter=',')
