import os
import torch
import random
import math
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, Sampler

from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader, Dataset_Physio, Dataset_PEMS, Dataset_Epilepsy, CIDatasetBenchmark,Dataset_Custom1,\
    Dataset_Solar





data_dict_text = {
    'ETTh1': Dataset_Custom1,
    'ETTh2': Dataset_Custom1,
    'ETTm1': Dataset_Custom1,
    'ETTm2': Dataset_Custom1,
    'Exchange': Dataset_Custom1,
    'Energy':Dataset_Custom1,
    'Health':Dataset_Custom1,
    'Environment':Dataset_Custom1,
    'Solar':Dataset_Custom1,
    'Wind':Dataset_Custom1,
}

data_path_dict={
    'ETTh1': "ETTh1.csv",
    'ETTh2': "ETTh2.csv",
    'ETTm1': "ETTm1.csv",
    'ETTm2': "ETTm2.csv",
    'Electricity': "electricity.csv",
    'Traffic': "traffic.csv",
    'Exchange': "exchange_rate.csv",
    'Weather': "weather.csv",
    'PEMS03': "PEMS03.npz",
    'PEMS04': "PEMS04.npz",
    'PEMS07': "PEMS07.npz",
    'PEMS08': "PEMS08.npz",
}

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Electricity': Dataset_Custom,
    'Traffic': Dataset_Custom,
    'Exchange': Dataset_Custom,
    'Weather': Dataset_Custom,
    'ECL': Dataset_Custom,
    'ILI': Dataset_Custom,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader,
    'HAR': Dataset_Physio,
    'EEG': Dataset_Physio,
    'PEMS03': Dataset_PEMS,
    'PEMS04': Dataset_PEMS,
    'PEMS07': Dataset_PEMS,
    'PEMS08': Dataset_PEMS,
    'Epilepsy': Dataset_Epilepsy,
    'Climate':Dataset_Custom1,
    'Energy':Dataset_Custom,
    'Health':Dataset_Custom,
    'SocialGood':Dataset_Custom,
    'Environment':Dataset_Custom,
    'Nature':Dataset_Custom,
    'Stock-NA':Dataset_Custom,
    'Stock-NY':Dataset_Custom,
    'Solar':Dataset_Solar,
    'Wind':Dataset_Custom,
}


def data_provider(args, flag,llm_model=None, tokenizer=None):

    timeenc = 0 if args.embed != 'timeF' else 1
    freq = args.freq
    if flag == 'test':
        batch_size = args.batch_size*2
        shuffle_flag = False
        drop_last = False
    elif flag == 'val':
        batch_size = args.batch_size*2   
        shuffle_flag = False
        drop_last = True
    else:
        batch_size = args.batch_size
        shuffle_flag = True
        drop_last = True



    Data = data_dict_text[args.data]
        
    if args.downstream_task == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            win_size=args.input_len,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )
        return data_set, data_loader
    elif args.downstream_task == 'classification':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            
        )
        return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False
        if llm_model:
            data_set = Data(
                root_path=args.root_path,
                data_path=args.data_path,
                flag=flag,
                size=[args.input_len, args.label_len, args.pred_len],
                features=args.features,
                target=args.target,
                timeenc=timeenc,
                freq=freq,
                seasonal_patterns=args.seasonal_patterns,
                llm_model=llm_model,          
                tokenizer=tokenizer,          
            )
        else:
            data_set = Data(
                root_path=args.root_path,
                data_path=args.data_path,
                flag=flag,
                size=[args.input_len, args.label_len, args.pred_len],
                features=args.features,
                target=args.target,
                timeenc=timeenc,
                freq=freq,
                seasonal_patterns=args.seasonal_patterns,
            )
            
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        print(flag, len(data_set), len(data_loader))
        return data_set, data_loader
