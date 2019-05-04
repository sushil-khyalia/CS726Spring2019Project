import numpy as np
import math
import torch


def masked_mse_torch(preds, labels, null_val=float('nan')):
    if np.isnan(null_val):
        mask = ~np.isnan(labels)
    else:
        mask = np.not_equal(labels, null_val)
    mask = mask.astype(np.float32)
    mask = torch.from_numpy(mask).type(torch.FloatTensor)
    if torch.mean(mask) != 0.0:
        mask /= torch.mean(mask)
    else:
        print("All values Nan in Labels")
        sys.exit()
    labels = torch.from_numpy(labels).type(torch.FloatTensor)
    mse = torch.square(preds - labels).type(torch.FloatTensor)
    mse = mse * mask
    return torch.mean(mse)

def masked_rmse_torch(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse_np(preds=preds, labels=labels, null_val=null_val))

def masked_mae_torch(preds, labels, null_val=float('nan')):
    if np.isnan(null_val):
        mask = ~np.isnan(labels)
    else:
        mask = np.not_equal(labels, null_val)
    mask = mask.astype(np.float32)
    mask = torch.from_numpy(mask).type(torch.FloatTensor)
    if torch.mean(mask) != 0:
        mask /= torch.mean(mask)
    else:
        print("All values Nan in Labels")
        sys.exit()
    labels = torch.from_numpy(labels).type(torch.FloatTensor)
    mae = torch.abs(preds - labels).type(torch.FloatTensor)
    mae = mae * mask
    return torch.mean(mae)


def masked_mape_torch(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~np.isnan(labels)
    else:
        mask = np.not_equal(labels, null_val)
    mask = mask.astype(np.float32)
    mask = torch.from_numpy(mask).type(torch.FloatTensor)
    if torch.mean(mask) != 0:
        mask /= torch.mean(mask)
    else:
        print("All values Nan in Labels")
        sys.exit()
    labels = torch.from_numpy(labels).type(torch.FloatTensor)
    mape = torch.abs((preds - labels)/labels).type(torch.FloatTensor)
    mape = mape * mask
    return torch.mean(mape)

def masked_rmse_np(preds, labels, null_val=np.nan):
    return np.sqrt(masked_mse_np(preds=preds, labels=labels, null_val=null_val))


def masked_mse_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        rmse = np.square(np.subtract(preds, labels)).astype('float32')
        rmse = np.nan_to_num(rmse * mask)
        return np.mean(rmse)


def masked_mae_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(preds, labels)).astype('float32')
        mae = np.nan_to_num(mae * mask)
        return np.mean(mae)


def masked_mape_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(preds, labels).astype('float32'), labels))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape)


# Builds loss function.
def masked_mse_loss(scaler, null_val):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        return masked_mse_torch(preds=preds, labels=labels, null_val=null_val)

    return loss


def masked_rmse_loss(scaler, null_val):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        return masked_rmse_torch(preds=preds, labels=labels, null_val=null_val)

    return loss


def masked_mae_loss(scaler, null_val):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        mae = masked_mae_torch(preds=preds, labels=labels, null_val=null_val)
        return mae

    return loss


def calculate_metrics(df_pred, df_test, null_val):
    """
    Calculate the MAE, MAPE, RMSE
    :param df_pred:
    :param df_test:
    :param null_val:
    :return:
    """
    mape = masked_mape_np(preds=df_pred.as_matrix(), labels=df_test.as_matrix(), null_val=null_val)
    mae = masked_mae_np(preds=df_pred.as_matrix(), labels=df_test.as_matrix(), null_val=null_val)
    rmse = masked_rmse_np(preds=df_pred.as_matrix(), labels=df_test.as_matrix(), null_val=null_val)
    return mae, mape, rmse


