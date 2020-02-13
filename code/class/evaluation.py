import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix


class Evaluation:
    """ This is a class with goals to call all methods that aim to perform evaluation for both models.
        """
    def plot_confusion_matrix_norm_multi_class(self,y_test, y_pred):
        '''
        this method has the purpose to plot the confusion matrix for multi-class classification problem. It takes as parameter the
        the targets from testing set and the target predicted by model.
        '''
        array = confusion_matrix(y_test, y_pred)
        print('Confusion Matrix Multi-class classification task: \n{}'.format(array))
        df_cm = pd.DataFrame(array, index=[i for i in ['HAZE', 'RAINY', 'SNOWY', 'SUNNY']],
                             columns=[i for i in ['HAZE', 'RAINY', 'SNOWY', 'SUNNY']])
        sn.set(font_scale=1.3)
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 20})  # font size
        plt.title('Multi-Class Classification Confusion Matrix')
        plt.show()
    def evaluation_errors(self,Ytest, Ypred, datagen, classnames):
        cm = confusion_matrix(Ytest, Ypred)

        conf = []  # data structure for confusions: list of (i,j,cm[i][j])
        for i in range(0, cm.shape[0]):
            for j in range(0, cm.shape[1]):
                if (i != j and cm[i][j] > 0):
                    conf.append([i, j, cm[i][j]])

        col = 2
        conf = np.array(conf)
        conf = conf[np.argsort(-conf[:, col])]  # decreasing order by 3-rd column (i.e., cm[i][j])

        print('%-16s     %-16s  \t%s \t%s ' % ('True', 'Predicted', 'errors', 'err %'))
        print('------------------------------------------------------------------')
        for k in conf:
            print('%-16s ->  %-16s  \t%d \t%.2f %% ' % (
                classnames[k[0]], classnames[k[1]], k[2], k[2] * 100.0 / datagen.n))