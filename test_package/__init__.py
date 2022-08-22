import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score

# Ridge, Lasso, ElasticNet의 MSE, RMSE 스코어 계산하는 함수

def score_checker(type, data, target, alpha):
    if type == 'ridge':
        type = Ridge(alpha)
        mse_score = cross_val_score(type, data, target, scoring="neg_mean_squared_error")
        rmse_score = np.sqrt(-1 * mse_score)
        avg_rmse = np.mean(rmse_score)

        print(f'Alpha : {alpha}')
        print(f'Ridge Negative MSE score : {np.abs(np.round(mse_score, 3))}')
        print(f'Ridge RMSE scores : {np.round(rmse_score, 3)}')
        print(f'Ridge AVG RMSE : {avg_rmse:.3f}\n')

    elif type == 'lasso':
        type = Lasso(alpha)
        mse_score = cross_val_score(type, data, target, scoring="neg_mean_squared_error")
        rmse_score = np.sqrt(-1 * mse_score)
        avg_rmse = np.mean(rmse_score)

        print(f'Alpha : {alpha}')
        print(f'Lasso Negative MSE score : {np.abs(np.round(mse_score, 3))}')
        print(f'Lasso RMSE scores : {np.round(rmse_score, 3)}')
        print(f'Lasso AVG RMSE : {avg_rmse:.3f}\n')

    elif type == 'elastic':
        type = ElasticNet(alpha)
        mse_score = cross_val_score(type, data, target, scoring="neg_mean_squared_error")
        rmse_score = np.sqrt(-1 * mse_score)
        avg_rmse = np.mean(rmse_score)

        print(f'Alpha : {alpha}')
        print(f'Elastic Negative MSE score : {np.abs(np.round(mse_score, 3))}')
        print(f'Elastic RMSE scores : {np.round(rmse_score, 3)}')
        print(f'Elastic AVG RMSE : {avg_rmse:.3f}')

    else:
        print(f'Check the values')