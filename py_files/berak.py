from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
import random

def berak_trend_line(data, trashhold = 8):
    sk_lr = LinearRegression()

    curr_y_list = []
    curr_X_list = []
    warn_points = []
    warn_index = []
    section_param = {}
    cur_pred_x = []
    treshhold = 2
    for index, point in enumerate(data):
        cur_pred_x.clear()
        if len(curr_y_list)<2:
            curr_y_list.append(point)
            curr_X_list.append(index)
        else:
            X=np.array(curr_X_list).reshape((-1, 1))
            y=np.array(curr_y_list)
            sk_lr.fit(X,y)
            cur_pred_x.append(index)
            cur_x = np.array(cur_pred_x).reshape((-1, 1))
            if abs(point-sk_lr.predict(cur_x)) >= trashhold:
                section_param[index] = {'x': tuple(curr_X_list), 'y':tuple(curr_y_list), 'k':sk_lr.coef_[0], 'b':sk_lr.intercept_}
                warn_points.append(point)
                warn_index.append(index)
                curr_y_list.clear()
                curr_X_list.clear()
                curr_y_list.append(point)
                curr_X_list.append(index)
            else:
                curr_y_list.append(point)
                curr_X_list.append(index)
    X=np.array(curr_X_list).reshape((-1, 1))
    y=np.array(curr_y_list)
    sk_lr.fit(X,y)
    section_param[index] = {'x': tuple(curr_X_list), 'y':tuple(curr_y_list), 'k':sk_lr.coef_[0], 'b':sk_lr.intercept_}
    return warn_index, warn_points, section_param