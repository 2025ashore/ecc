# -*- coding: utf-8 -*-
"""
ECG数据集处理核心模块
功能：实现PTB-XL/CPSC/HF三大ECG数据集的加载、预处理、标签处理全流程
适配多标签分类任务，支持数据缓存、采样率切换、标签聚合、信号标准化等关键操作
"""
import os
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm  # 进度条库，用于显示数据加载进度
import wfdb  # 医学信号处理库，用于读取ECG的.dat格式文件
import ast  # 字符串转字典工具，用于解析PTB-XL/CPSC的标签字段
from scipy.signal import resample  # 信号重采样，统一ECG信号长度
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer  # 标准化、多标签编码
from sklearn import preprocessing  # 预处理工具（含MinMaxScaler）


# -----------------------------------------------------------------------------
# 核心函数：数据集加载（自动适配PTB-XL/CPSC）
# -----------------------------------------------------------------------------
def load_dataset(path, sampling_rate, release=False):
    """
    加载PTB-XL或CPSC数据集（根据路径自动判断）
    Args:
        path: 数据集根路径（如 'data/ptbxl/' 或 'data/CPSC/'）
        sampling_rate: 采样率（100或500Hz，ECG常用采样率）
        release: 是否为公开版本（预留参数，暂未使用）
    Returns:
        X: ECG信号数组，形状为 [样本数, 时间步长, 导联数]（默认12导联）
        Y: 标签数据框，包含原始标签（scp_codes）及后续处理的标签字段
    """
    # 根据路径最后一级目录判断数据集类型（ptbxl/CPSC）
    if path.split('/')[-2] == 'ptbxl':
        # 加载PTB-XL的标签文件（核心元数据，含每个样本的信号路径和标签）
        Y = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')
        # 将scp_codes列从字符串（如"{'MI':1, 'NORM':0}"）转为字典格式
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
        # 加载PTB-XL的原始ECG信号
        X = load_raw_data_ptbxl(Y, sampling_rate, path)

    elif path.split('/')[-2] == 'CPSC':
        # 加载CPSC的标签文件（格式与PTB-XL类似）
        Y = pd.read_csv(path + 'cpsc_database.csv', index_col='ecg_id')
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
        # 加载CPSC的原始ECG信号
        X = load_raw_data_cpsc(Y, sampling_rate, path)

    return X, Y


# -----------------------------------------------------------------------------
# 辅助函数：加载CPSC数据集原始信号
# -----------------------------------------------------------------------------
def load_raw_data_cpsc(df, sampling_rate, path):
    """
    加载CPSC数据集的原始ECG信号（支持100/500Hz采样率，带缓存机制）
    Args:
        df: 标签数据框（含样本索引，用于匹配信号文件）
        sampling_rate: 目标采样率（100或500）
        path: CPSC数据集根路径
    Returns:
        data: 标准化后的ECG信号数组 [样本数, 时间步长, 12]
    """
    # 100Hz采样率（信号文件存放在records100目录）
    if sampling_rate == 100:
        # 缓存机制：若已生成.npy文件，直接加载（避免重复读取.dat，提升速度）
        if os.path.exists(path + 'raw100.npy'):
            data = np.load(path + 'raw100.npy', allow_pickle=True)
        else:
            # 遍历所有样本，用wfdb读取.dat信号文件（tqdm显示进度）
            data = [wfdb.rdsamp(path + 'records100/' + str(f)) for f in tqdm(df.index)]
            # 提取信号部分（wfdb返回格式为 (signal, meta)，取signal）
            data = np.array([signal for signal, meta in data])
            # 保存为.npy文件，后续直接复用
            pickle.dump(data, open(path + 'raw100.npy', 'wb'), protocol=4)

    # 500Hz采样率（信号文件存放在records500目录，逻辑同上）
    elif sampling_rate == 500:
        if os.path.exists(path + 'raw500.npy'):
            data = np.load(path + 'raw500.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path + 'records500/' + str(f)) for f in tqdm(df.index)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(path + 'raw500.npy', 'wb'), protocol=4)

    return data


# -----------------------------------------------------------------------------
# 辅助函数：加载PTB-XL数据集原始信号
# -----------------------------------------------------------------------------
def load_raw_data_ptbxl(df, sampling_rate, path):
    """
    加载PTB-XL数据集的原始ECG信号（支持100/500Hz采样率，带缓存机制）
    与CPSC的区别：PTB-XL的信号路径存放在df.filename_lr（100Hz）/filename_hr（500Hz）中
    Args:
        df: 标签数据框（含filename_lr/filename_hr字段，指向信号文件）
        sampling_rate: 目标采样率（100或500）
        path: PTB-XL数据集根路径
    Returns:
        data: 标准化后的ECG信号数组 [样本数, 时间步长, 12]
    """
    if sampling_rate == 100:
        if os.path.exists(path + 'raw100.npy'):
            data = np.load(path + 'raw100.npy', allow_pickle=True)
        else:
            # 从df.filename_lr获取100Hz信号路径，拼接根路径后读取
            data = [wfdb.rdsamp(path + f) for f in tqdm(df.filename_lr)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(path + 'raw100.npy', 'wb'), protocol=4)

    elif sampling_rate == 500:
        if os.path.exists(path + 'raw500.npy'):
            data = np.load(path + 'raw500.npy', allow_pickle=True)
        else:
            # 从df.filename_hr获取500Hz信号路径
            data = [wfdb.rdsamp(path + f) for f in tqdm(df.filename_hr)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(path + 'raw500.npy', 'wb'), protocol=4)

    return data


# -----------------------------------------------------------------------------
# 核心函数：标签聚合（将原始SCP代码映射为不同层级的标签）
# -----------------------------------------------------------------------------
def compute_label_aggregations(df, folder, ctype):
    """
    标签聚合：将原始SCP代码（如"MI"、"STTC"）映射为不同层级的标签（诊断类、节律类等）
    依赖scp_statements.csv文件（SCP代码的分类定义）
    Args:
        df: 含scp_codes字段的数据框
        folder: 数据集根路径（用于读取scp_statements.csv）
        ctype: 标签类型（控制聚合方式）
            - diagnostic: 原始诊断标签（保留所有SCP代码）
            - subdiagnostic: 子诊断层级标签
            - superdiagnostic: 超级诊断层级标签（最高抽象层级）
            - form: 波形形态标签
            - rhythm: 节律标签
            - all: 所有SCP代码
    Returns:
        df: 新增聚合后标签字段的数据框（如diagnostic、rhythm等）
    """
    # 统计每个样本的原始标签数量（用于过滤无标签样本）
    df['scp_codes_len'] = df.scp_codes.apply(lambda x: len(x))

    # -------- CPSC 数据集：标签已在 convert_cpsc.py 中直接生成，无需 scp_statements.csv --------
    # CPSC 的 scp_codes 已经是最终标签（如 {'NORM':100, 'AFIB':100}），直接提取 key 即可
    if folder.rstrip('/').endswith('CPSC'):
        df['all_scp'] = df.scp_codes.apply(lambda x: list(set(x.keys())))
        df['all_scp_len'] = df['all_scp'].apply(lambda x: len(x))
        return df

    # 加载SCP代码分类定义文件（核心映射表，仅 PTB-XL 需要）
    aggregation_df = pd.read_csv(folder + 'scp_statements.csv', index_col=0)

    # -------------------------- 诊断相关标签聚合 --------------------------
    if ctype in ['diagnostic', 'subdiagnostic', 'superdiagnostic']:
        # 筛选出诊断类的SCP代码（scp_statements.csv中diagnostic=1.0的行）
        diag_agg_df = aggregation_df[aggregation_df.diagnostic == 1.0]

        # 聚合所有原始诊断标签（保留所有SCP代码）
        def aggregate_all_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    tmp.append(key)
            return list(set(tmp))  # 去重

        # 聚合子诊断层级标签（映射到subclass）
        def aggregate_subdiagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    c = diag_agg_df.loc[key].diagnostic_subclass
                    if str(c) != 'nan':  # 过滤空值
                        tmp.append(c)
            return list(set(tmp))

        # 聚合超级诊断层级标签（映射到class，最高层级）
        def aggregate_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    c = diag_agg_df.loc[key].diagnostic_class
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        # 根据ctype添加对应的聚合标签列
        if ctype == 'diagnostic':
            df['diagnostic'] = df.scp_codes.apply(aggregate_all_diagnostic)
            df['diagnostic_len'] = df.diagnostic.apply(lambda x: len(x))  # 标签数量
        elif ctype == 'subdiagnostic':
            df['subdiagnostic'] = df.scp_codes.apply(aggregate_subdiagnostic)
            df['subdiagnostic_len'] = df.subdiagnostic.apply(lambda x: len(x))
        elif ctype == 'superdiagnostic':
            df['superdiagnostic'] = df.scp_codes.apply(aggregate_diagnostic)
            df['superdiagnostic_len'] = df.superdiagnostic.apply(lambda x: len(x))

    # -------------------------- 波形形态标签聚合 --------------------------
    elif ctype == 'form':
        # 筛选形态类SCP代码（form=1.0）
        form_agg_df = aggregation_df[aggregation_df.form == 1.0]

        def aggregate_form(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in form_agg_df.index:
                    tmp.append(key)
            return list(set(tmp))

        df['form'] = df.scp_codes.apply(aggregate_form)
        df['form_len'] = df.form.apply(lambda x: len(x))

    # -------------------------- 节律标签聚合 --------------------------
    elif ctype == 'rhythm':
        # 筛选节律类SCP代码（rhythm=1.0）
        rhythm_agg_df = aggregation_df[aggregation_df.rhythm == 1.0]

        def aggregate_rhythm(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in rhythm_agg_df.index:
                    tmp.append(key)
            return list(set(tmp))

        df['rhythm'] = df.scp_codes.apply(aggregate_rhythm)
        df['rhythm_len'] = df.rhythm.apply(lambda x: len(x))

    # -------------------------- 所有SCP代码聚合 --------------------------
    elif ctype == 'all':
        df['all_scp'] = df.scp_codes.apply(lambda x: list(set(x.keys())))  # 去重后保留所有代码
        # 修复：添加标签长度列（匹配其他ctype的格式）
        df['all_scp_len'] = df['all_scp'].apply(lambda x: len(x))

    return df


# -----------------------------------------------------------------------------
# 核心函数：数据筛选与多标签编码
# -----------------------------------------------------------------------------
def select_data(XX, YY, ctype, min_samples):
    """
    筛选有效样本（过滤无标签/标签过少的样本）并将多标签转为多热编码
    Args:
        XX: 原始ECG信号数组
        YY: 含聚合标签的数据框
        ctype: 标签类型（与compute_label_aggregations的ctype对应）
        min_samples: 最小样本数阈值（过滤样本数少于该值的标签）
    Returns:
        X: 筛选后的信号数组
        Y: 筛选后的标签数据框
        y: 多热编码后的标签矩阵 [样本数, 类别数]（1表示存在该标签，0表示不存在）
        mlb: MultiLabelBinarizer实例（用于后续解码标签）
    """
    mlb = MultiLabelBinarizer()  # 多标签编码工具

    # 根据标签类型筛选数据
    if ctype == 'diagnostic':
        # 过滤无诊断标签的样本（diagnostic_len>0）
        X = XX[YY.diagnostic_len > 0]
        Y = YY[YY.diagnostic_len > 0]
        # 对诊断标签进行多热编码
        mlb.fit(Y.diagnostic.values)
        y = mlb.transform(Y.diagnostic.values)

    elif ctype == 'subdiagnostic':
        # 过滤样本数少于min_samples的标签
        counts = pd.Series(np.concatenate(YY.subdiagnostic.values)).value_counts()
        counts = counts[counts > min_samples]
        # 只保留高频标签
        YY.subdiagnostic = YY.subdiagnostic.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['subdiagnostic_len'] = YY.subdiagnostic.apply(lambda x: len(x))
        # 过滤无有效标签的样本
        X = XX[YY.subdiagnostic_len > 0]
        Y = YY[YY.subdiagnostic_len > 0]
        mlb.fit(Y.subdiagnostic.values)
        y = mlb.transform(Y.subdiagnostic.values)

    elif ctype == 'superdiagnostic':
        # 逻辑同subdiagnostic，针对超级诊断标签
        counts = pd.Series(np.concatenate(YY.superdiagnostic.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.superdiagnostic = YY.superdiagnostic.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['superdiagnostic_len'] = YY.superdiagnostic.apply(lambda x: len(x))
        X = XX[YY.superdiagnostic_len > 0]
        Y = YY[YY.superdiagnostic_len > 0]
        mlb.fit(Y.superdiagnostic.values)
        y = mlb.transform(Y.superdiagnostic.values)

    elif ctype == 'form' or ctype == 'rhythm' or ctype == 'all':
        # 修复：根据ctype映射正确的列名（ctype='all'时对应'all_scp'列）
        col_name = 'all_scp' if ctype == 'all' else ctype
        len_col_name = f'{col_name}_len'  # 长度列名（form_len/rhythm_len/all_scp_len）

        # 统计标签出现次数，过滤低频标签
        counts = pd.Series(np.concatenate(YY[col_name].values)).value_counts()
        counts = counts[counts > min_samples]

        # 只保留高频标签
        YY[col_name] = YY[col_name].apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY[len_col_name] = YY[col_name].apply(lambda x: len(x))

        # 过滤无有效标签的样本
        X = XX[YY[len_col_name] > 0]
        Y = YY[YY[len_col_name] > 0]

        # 多标签编码
        mlb.fit(Y[col_name].values)
        y = mlb.transform(Y[col_name].values)

    else:
        X, Y, y, mlb = None, None, None, None  # 无效ctype返回空

    return X, Y, y, mlb


# -----------------------------------------------------------------------------
# 核心函数：信号标准化（均值0，方差1）
# -----------------------------------------------------------------------------
def preprocess_signals(X_train, X_validation, X_test):
    """
    对训练/验证/测试集的ECG信号进行标准化（基于训练集统计量，避免数据泄露）
    Args:
        X_train: 训练集信号 [N1, T, 12]
        X_validation: 验证集信号 [N2, T, 12]
        X_test: 测试集信号 [N3, T, 12]
    Returns:
        标准化后的训练/验证/测试集信号
    """
    ss = StandardScaler()  # 标准化工具（mean=0, std=1）
    # 基于训练集所有信号的统计量拟合标准化器（避免测试集信息泄露）
    ss.fit(np.vstack(X_train).flatten()[:, np.newaxis].astype(float))
    # 对三个数据集分别应用标准化
    return apply_standardizer(X_train, ss), apply_standardizer(X_validation, ss), apply_standardizer(X_test, ss)


def apply_standardizer(X, ss):
    """
    对单个数据集应用标准化（辅助函数）
    Args:
        X: 信号数组 [N, T, 12]
        ss: 已拟合的StandardScaler实例
    Returns:
        X_tmp: 标准化后的信号数组
    """
    X_tmp = []
    for x in X:
        x_shape = x.shape  # 保存原始形状 [T, 12]
        # 展平后标准化 → 恢复原始形状
        X_tmp.append(ss.transform(x.flatten()[:, np.newaxis]).reshape(x_shape))
    return np.array(X_tmp)


# -----------------------------------------------------------------------------
# 辅助函数：信号长度统一（固定为1000个时间步）
# -----------------------------------------------------------------------------
def data_slice(data):
    """
    统一ECG信号长度为1000个时间步，确保输入模型的维度一致
    - 小于1000：重采样到1000（避免补零导致的噪声）
    - 大于1000：截取前1000个时间步
    - 导联数修正：确保为12导联（丢弃多余导联）
    Args:
        data: 原始信号数组 [N, T, C]（C为导联数）
    Returns:
        data_process: 长度统一后的信号数组 [N, 1000, 12]
    """
    data_process = []
    for dat in data:
        # 统一时间步长为1000
        if dat.shape[0] < 1000:
            dat = resample(dat, 1000, axis=0)  # 重采样
        elif dat.shape[0] > 1000:
            dat = dat[:1000, :]  # 截取前1000个点
        # 统一导联数为12（丢弃多余导联）
        if dat.shape[1] != 12:
            dat = dat[:, 0:12]
        data_process.append(dat)
    return np.array(data_process)


# -----------------------------------------------------------------------------
# HF数据集专用处理函数（用户已说明无法下载，仅保留注释供参考）
# -----------------------------------------------------------------------------
def name2index(path):
    """
    HF数据集：将心律失常名称映射为索引（如"窦性心律"→0）
    依赖hf_round2_arrythmia.txt文件（HF数据集的标签名称列表）
    Args:
        path: hf_round2_arrythmia.txt文件路径
    Returns:
        name2indx: 名称→索引的字典
    """
    list_name = []
    for line in open(path, encoding='utf-8'):
        list_name.append(line.strip())  # 读取所有标签名称
    name2indx = {name: i for i, name in enumerate(list_name)}  # 名称映射为索引
    return name2indx


def file2index(path, name2idx):
    """
    HF数据集：将每个样本的标签名称转为索引（多标签）
    依赖hf_round2_label.txt文件（样本ID→标签名称的映射）
    Args:
        path: hf_round2_label.txt文件路径
        name2idx: 名称→索引的字典（来自name2index函数）
    Returns:
        file2index: 样本ID→标签索引列表的字典
    """
    file2index = dict()
    for line in open(path, encoding='utf-8'):
        arr = line.strip().split('\t')
        id = arr[0]  # 样本ID
        labels = [name2idx[name] for name in arr[3:]]  # 标签名称转索引
        file2index[id] = labels
    return file2index


def load_raw_data_hf(root='data/hf/', resample_num=1000, num_classes=34):
    """
    HF数据集：加载原始信号并转换为模型输入格式（带缓存机制）
    信号处理逻辑：读取CSV格式信号→构造12导联→重采样→MinMax归一化
    Args:
        root: HF数据集根路径
        resample_num: 目标时间步长（默认1000）
        num_classes: 标签类别数（默认34）
    Returns:
        data: 处理后的ECG信号 [N, 1000, 12]
        y: 多热编码后的标签 [N, 34]
    """
    # 缓存机制：已生成缓存文件则直接加载
    if os.path.exists(root + 'raw100_data.npy'):
        data = np.load(root + 'raw100_data.npy', allow_pickle=True)
        y = np.load(root + 'raw100_label.npy', allow_pickle=True)
    else:
        # 标签名称→索引映射
        name2idx = name2index(root + 'hf_round2_arrythmia.txt')
        # 样本ID→标签索引映射
        file2idx = file2index(root + 'hf_round2_label.txt', name2idx)
        data, label = [], []
        # 遍历所有样本文件
        for file, list_idx in file2idx.items():
            temp = np.zeros([5000, 12])  # 初始化5000时间步×12导联的信号矩阵
            # 读取HF的CSV格式信号（与PTB-XL/CPSC的.dat格式不同）
            df = pd.read_csv(root + 'hf_round2_train' + '/' + file, sep=' ').values
            # 构造12导联信号（HF原始信号非12导联，需手动映射）
            temp[:, 2] = df[:, 1] - df[:, 0]
            temp[:, 3] = -(df[:, 0] + df[:, 1]) / 2
            temp[:, 4] = df[:, 0] - df[:, 1] / 2
            temp[:, 5] = df[:, 1] - df[:, 0] / 2
            temp[:, 0:2] = df[:, 0:2]
            temp[:, 6:12] = df[:, 2:8]
            # 重采样到目标时间步长
            sig = resample(temp, resample_num)
            # MinMax归一化（映射到[0,1]）
            min_max_scaler = preprocessing.MinMaxScaler()
            ecg = min_max_scaler.fit_transform(sig)
            data.append(ecg)
            label.append(tuple(list_idx))  # 标签索引转为元组（适配MultiLabelBinarizer）
        # 转为数组并保存缓存
        data = np.array(data)
        pickle.dump(data, open(root + 'raw100_data.npy', 'wb'), protocol=4)
        # 多标签编码
        mlb = MultiLabelBinarizer(classes=[i for i in range(num_classes)])
        y = mlb.fit_transform(label)
        y = np.array(y)
        pickle.dump(y, open(root + 'raw100_label.npy', 'wb'), protocol=4)
    return data, y


def hf_dataset(root='data/hf/', resample_num=1000, num_classes=34):
    """
    HF数据集：划分训练/验证/测试集（2:6:2比例）
    Args:
        root: HF数据集根路径
        resample_num: 目标时间步长
        num_classes: 标签类别数
    Returns:
        X_train/y_train: 训练集（60%）
        X_val/y_val: 验证集（20%）
        X_test/y_test: 测试集（20%）
    """
    # 加载处理后的信号和标签
    data, label = load_raw_data_hf(root, resample_num, num_classes)
    data_num = len(label)
    # 随机打乱数据
    shuffle_ix = np.random.permutation(np.arange(data_num))
    data = data[shuffle_ix]
    labels = label[shuffle_ix]

    # 划分比例：20%测试集 → 60%训练集 → 20%验证集
    X_train = data[int(data_num * 0.2):int(data_num * 0.8)]
    y_train = labels[int(data_num * 0.2):int(data_num * 0.8)]
    X_val = data[int(data_num * 0.8):]
    y_val = labels[int(data_num * 0.8):]
    X_test = data[:int(data_num * 0.2)]
    y_test = labels[:int(data_num * 0.2)]

    return X_train, y_train, X_val, y_val, X_test, y_test