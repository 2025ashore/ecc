import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
# 导入自定义的数据处理工具函数（加载数据集、标签聚合、数据选择、信号预处理、数据切片、HF数据集加载）
from data_process import load_dataset, compute_label_aggregations, select_data, preprocess_signals, data_slice, \
    hf_dataset
# 导入配置文件（包含批大小等训练参数）
from config import config


class ECGDataset(Dataset):
    """
    PyTorch标准数据集类，用于封装ECG信号数据和对应的标签
    适配PyTorch的DataLoader，支持批量加载数据
    """

    def __init__(self, signals: np.ndarray, labels: np.ndarray):
        """
        初始化ECGDataset实例

        参数说明：
            signals: np.ndarray - ECG信号数据，形状为[样本数, 时间步长, 导联数]
            labels: np.ndarray - 标签数据，形状为[样本数, 类别数]（独热编码格式）
        """
        super(ECGDataset, self).__init__()
        self.data = signals  # 存储ECG信号数据
        self.label = labels  # 存储对应标签
        self.num_classes = self.label.shape[1]  # 计算类别数量（标签的列数）

        # 统计每个类别的样本数量（按列求和，因为标签是独热编码）
        self.cls_num_list = np.sum(self.label, axis=0)

    def __getitem__(self, index):
        """
        按索引获取单个样本（PyTorch Dataset核心方法）

        参数：
            index: int - 样本索引
        返回：
            x: torch.Tensor - 单个ECG信号张量，形状为[导联数, 时间步长]
            y: torch.Tensor - 单个样本标签张量，形状为[类别数]
        """
        # 获取索引对应的信号和标签
        x = self.data[index]
        y = self.label[index]

        # 转置信号维度：从[时间步长, 导联数]转为[导联数, 时间步长]
        # 适配PyTorch中卷积层输入格式（通道数在前，时间/空间维度在后）
        x = x.transpose()

        # 转换为torch张量（float类型），copy()避免numpy数组与张量共享内存导致的潜在问题
        x = torch.tensor(x.copy(), dtype=torch.float)

        # 转换标签为torch.float张量，squeeze()去除多余维度（确保形状为[类别数]）
        y = torch.tensor(y, dtype=torch.float)
        y = y.squeeze()

        return x, y

    def __len__(self):
        """
        返回数据集总样本数（PyTorch Dataset核心方法）
        """
        return len(self.data)


class DownLoadECGData:
    '''
    数据集下载与预处理主类，支持PTB-XL和CPSC数据集
    负责加载原始数据、处理标签、划分训练/验证/测试集、信号预处理
    '''

    def __init__(self, experiment_name, task, datafolder, sampling_frequency=100, min_samples=0,
                 train_fold=8, val_fold=9, test_fold=10):
        """
        初始化数据处理实例

        参数说明：
            experiment_name: str - 实验名称（用于标识不同实验配置）
            task: str - 任务类型（如'all'/'diagnostic'/'rhythm'，对应不同标签聚合方式）
            datafolder: str - 数据集存储根路径
            sampling_frequency: int - 信号采样频率（默认100Hz，支持100/500Hz）
            min_samples: int - 最小样本数阈值（用于过滤样本量过少的类别，默认0不过滤）
            train_fold: int - 训练集对应的最大fold数（fold<=train_fold为训练集）
            val_fold: int - 验证集对应的fold数（fold==val_fold为验证集）
            test_fold: int - 测试集对应的fold数（fold==test_fold为测试集）
        """
        self.min_samples = min_samples  # 类别最小样本数阈值
        self.task = task  # 标签聚合任务类型
        self.train_fold = train_fold  # 训练集fold边界
        self.val_fold = val_fold  # 验证集fold编号
        self.test_fold = test_fold  # 测试集fold编号
        self.experiment_name = experiment_name  # 实验名称
        self.datafolder = datafolder  # 数据集路径
        self.sampling_frequency = sampling_frequency  # 采样频率

    def preprocess_data(self):
        """
        完整的数据预处理流程：加载数据→处理标签→选择数据→划分数据集→信号预处理

        返回：
            X_train: np.ndarray - 训练集信号数据
            y_train: np.ndarray - 训练集标签（独热编码）
            X_val: np.ndarray - 验证集信号数据
            y_val: np.ndarray - 验证集标签（独热编码）
            X_test: np.ndarray - 测试集信号数据
            y_test: np.ndarray - 测试集标签（独热编码）
        """
        # 1. 加载原始数据集（信号数据和原始标签）
        # load_dataset根据datafolder自动识别数据集类型（PTB-XL/CPSC），返回对应格式数据
        data, raw_labels = load_dataset(self.datafolder, self.sampling_frequency)

        # 2. 标签聚合处理（根据任务类型将原始标签映射为统一格式，如诊断类/节律类标签）
        labels = compute_label_aggregations(raw_labels, self.datafolder, self.task)

        # 3. 选择有效数据并转换为独热编码标签
        # select_data过滤样本量过少的类别，返回筛选后的信号、标签元数据、独热编码标签等
        data, labels, Y, _ = select_data(data, labels, self.task, self.min_samples)

        # 4. 针对CPSC数据集的特殊处理：信号切片（确保所有样本长度一致）
        if self.datafolder == 'data/CPSC/':
            data = data_slice(data)

        # 5. 按strat_fold划分训练/验证/测试集（分层划分，保证类别分布一致）
        X_test = data[labels.strat_fold == self.test_fold]  # 测试集：fold==test_fold
        y_test = Y[labels.strat_fold == self.test_fold]
        X_val = data[labels.strat_fold == self.val_fold]  # 验证集：fold==val_fold
        y_val = Y[labels.strat_fold == self.val_fold]
        X_train = data[labels.strat_fold <= self.train_fold]  # 训练集：fold<=train_fold
        y_train = Y[labels.strat_fold <= self.train_fold]

        # 6. 信号预处理（如归一化、去噪等，具体逻辑在preprocess_signals中实现）
        X_train, X_val, X_test = preprocess_signals(X_train, X_val, X_test)

        return X_train, y_train, X_val, y_val, X_test, y_test


def load_datasets(datafolder=None, experiment=None):
    '''
    数据集加载主函数：根据数据路径和实验配置，返回训练/验证/测试集的DataLoader

    参数说明：
        datafolder: str - 数据集存储根路径（如'data/ptbxl/'/'data/CPSC/'）
        experiment: str - 实验配置名称（仅PTB-XL需要，对应不同任务类型）

    返回：
        train_dataloader: DataLoader - 训练集数据加载器
        val_dataloader: DataLoader - 验证集数据加载器
        test_dataloader: DataLoader - 测试集数据加载器
        num_classes: int - 数据集类别数量
    '''
    experiment = experiment  # 实验配置名称（用于PTB-XL的多任务选择）

    # 分支1：处理PTB-XL数据集（支持多个实验任务）
    if datafolder == 'data/ptbxl/':
        # PTB-XL实验配置字典：key=实验名称，value=(实验标识, 任务类型)
        experiments = {
            'exp0': ('exp0', 'all'),  # 所有标签
            'exp1': ('exp1', 'diagnostic'),  # 诊断类标签
            'exp1.1': ('exp1.1', 'subdiagnostic'),  # 子诊断类标签
            'exp1.1.1': ('exp1.1.1', 'superdiagnostic'),  # 超诊断类标签
            'exp2': ('exp2', 'form'),  # 形态类标签
            'exp3': ('exp3', 'rhythm')  # 节律类标签
        }
        # 兼容消融实验名称：'exp0_abl_A' → 取基础实验名 'exp0' 用于数据加载
        base_experiment = experiment.split('_abl_')[0] if experiment and '_abl_' in experiment else experiment
        # 根据实验名称获取对应的实验标识和任务类型
        name, task = experiments[base_experiment]
        # 初始化数据处理实例
        ded = DownLoadECGData(name, task, datafolder)
        # 执行数据预处理，获取划分后的数据集
        X_train, y_train, X_val, y_val, X_test, y_test = ded.preprocess_data()

    # 分支2：处理CPSC数据集（固定任务类型为'all'）
    elif datafolder == 'data/CPSC/':
        ded = DownLoadECGData('exp_CPSC', 'all', datafolder)
        X_train, y_train, X_val, y_val, X_test, y_test = ded.preprocess_data()

    # 分支3：处理Hugging Face数据集（通过hf_dataset函数加载）
    else:
        X_train, y_train, X_val, y_val, X_test, y_test = hf_dataset(datafolder)

    # 封装为ECGDataset实例
    ds_train = ECGDataset(X_train, y_train)  # 训练集
    ds_val = ECGDataset(X_val, y_val)  # 验证集
    ds_test = ECGDataset(X_test, y_test)  # 测试集

    # 获取类别数量（从训练集获取，确保一致性）
    num_classes = ds_train.num_classes

    # 创建PyTorch DataLoader（批量加载数据）
    train_dataloader = DataLoader(ds_train, batch_size=config.batch_size, shuffle=True)  # 训练集打乱
    val_dataloader = DataLoader(ds_val, batch_size=config.batch_size, shuffle=False)  # 验证集不打乱
    test_dataloader = DataLoader(ds_test, batch_size=config.batch_size, shuffle=False)  # 测试集不打乱

    return train_dataloader, val_dataloader, test_dataloader, num_classes