# 导入所需库
import os  # 用于文件路径操作和目录创建
import pandas as pd  # 用于数据框处理（存储元信息、读取参考文件）
import wfdb  # 用于WFDB格式文件的读写（心电图标准格式）
from tqdm import tqdm  # 用于显示循环处理进度条
import numpy as np  # 用于数组运算和数据处理
from scipy.ndimage import zoom  # 用于信号下采样（调整采样率）
from scipy.io import loadmat  # 用于加载MAT格式的原始心电图数据
from stratisfy import stratisfy_df  # 用于数据集分层折叠（保证分布一致性）

# 定义输出文件夹路径
output_folder = 'data/CPSC/'  # 总输出目录（存放所有处理后的数据）
output_datafolder_100 = output_folder + '/records100/'  # 100Hz采样率ECG数据存储目录
output_datafolder_500 = output_folder + '/records500/'  # 500Hz采样率ECG数据存储目录

# 创建文件夹（若不存在）
if not os.path.exists(output_folder):
    os.mkdir(output_folder)  # 创建总目录
if not os.path.exists(output_datafolder_100):
    os.makedirs(output_datafolder_100)  # 创建100Hz数据目录
if not os.path.exists(output_datafolder_500):
    os.makedirs(output_datafolder_500)  # 创建500Hz数据目录


def store_as_wfdb(signame, data, sigfolder, fs):
    """
    将ECG信号保存为WFDB格式文件（包含.head和.dat文件）

    参数说明：
        signame: str - 信号唯一标识名称（对应ecg_id）
        data: np.ndarray - ECG信号数据，形状为[时间点数量, 通道数]（12导联）
        sigfolder: str - 保存目录路径
        fs: int - 采样率（Hz）
    """
    # 12导联心电图的标准通道名称映射（从索引到导联名）
    channel_itos = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    # 调用wfdb.wrsamp写入WFDB文件
    wfdb.wrsamp(
        record_name=signame,  # 记录名称（无后缀）
        fs=fs,  # 采样率
        sig_name=channel_itos,  # 各通道导联名称
        p_signal=data,  # 生理信号数据（浮点型）
        units=['mV'] * len(channel_itos),  # 所有通道的单位（毫伏）
        fmt=['16'] * len(channel_itos),  # 数据存储格式（16位整数，节省空间）
        write_dir=sigfolder  # 保存目录
    )


# 加载标签参考文件（包含每个ECG记录的疾病标签）
df_reference = pd.read_csv('data/CPSC/REFERENCE.csv')

# 标签编码与疾病名称映射字典（原始标签为数字编码，转换为标准名称）
label_dict = {
    1: 'NORM',  # 正常心电图
    2: 'AFIB',  # 心房颤动
    3: '1AVB',  # 一度房室传导阻滞
    4: 'CLBBB',  # 完全性左束支传导阻滞
    5: 'CRBBB',  # 完全性右束支传导阻滞
    6: 'PAC',  # 房性早搏
    7: 'VPC',  # 室性早搏
    8: 'STD_',  # ST段压低
    9: 'STE_'  # ST段抬高
}

# 初始化存储数据集元信息的字典（后续转为DataFrame）
data = {
    'ecg_id': [],  # ECG数据唯一标识（自增整数）
    'filename': [],  # 原始MAT文件名称（无.mat后缀）
    'validation': [],  # 是否为验证集（此处均为训练集，设为False）
    'age': [],  # 患者年龄
    'sex': [],  # 患者性别（1=男性，0=女性）
    'scp_codes': []  # 疾病标签字典（键：疾病名称，值：100表示存在该标签）
}

ecg_counter = 0  # ECG唯一标识计数器（从1开始）

# 遍历三个原始训练集文件夹（TrainingSet1/2/3存放原始MAT数据）
for folder in ['TrainingSet1', 'TrainingSet2', 'TrainingSet3']:
    # 获取当前文件夹下的所有文件名称
    filenames = os.listdir('data/CPSC/' + folder)
    # 遍历文件，tqdm显示处理进度
    for filename in tqdm(filenames, desc=f'Processing {folder}'):
        # 只处理MAT格式文件（原始ECG数据存储格式）
        if filename.split('.')[1] == 'mat':
            ecg_counter += 1  # 递增计数器，生成唯一ECG标识
            name = filename.split('.')[0]  # 获取文件名称（去掉.mat后缀）

            # 加载MAT文件并提取关键信息：sex（性别）、age（年龄）、sig（ECG信号）
            # MAT文件结构：ECG[0][0]包含三个元素：性别字符串、年龄数组、信号数组
            sex, age, sig = loadmat('data/CPSC/' + folder + '/' + filename)['ECG'][0][0]

            # 将信息添加到元数据字典中
            data['ecg_id'].append(ecg_counter)  # 添加唯一标识
            data['filename'].append(name)  # 添加原始文件名
            data['validation'].append(False)  # 标记为非验证集
            data['age'].append(age[0][0])  # 提取年龄（age为二维数组，取第一个元素）
            # 性别转换：Male→1，Female→0
            data['sex'].append(1 if sex[0] == 'Male' else 0)

            # 从参考文件中提取当前ECG的所有标签
            # 匹配Recording字段（与原始文件名一致），获取三个可能的标签列
            labels = df_reference[df_reference.Recording == name][
                ['First_label', 'Second_label', 'Third_label']].values.flatten()
            labels = labels[~np.isnan(labels)].astype(int)  # 过滤NaN值并转为整数编码

            # 转换标签编码为疾病名称，生成标签字典（值设为100表示标签权重）
            data['scp_codes'].append({label_dict[key]: 100 for key in labels})

            # 保存500Hz原始采样率的ECG数据（WFDB格式）
            # sig原始形状为[12, 时间点]，转置为[时间点, 12]以符合WFDB要求
            store_as_wfdb(str(ecg_counter), sig.T, output_datafolder_500, 500)

            # 下采样到100Hz（原始500Hz→100Hz，缩放比例0.2）
            # 对每个导联通道单独下采样，保持通道维度不变
            down_sig = np.array([zoom(channel, 0.2) for channel in sig])
            # 保存100Hz下采样后的ECG数据（WFDB格式）
            store_as_wfdb(str(ecg_counter), down_sig.T, output_datafolder_100, 100)

# 将元数据字典转为DataFrame
df = pd.DataFrame(data)
# 添加patient_id字段（与ecg_id一致，用于患者标识关联）
df['patient_id'] = df.ecg_id
# 分层折叠处理（基于标签分布，生成strat_fold字段，用于后续划分训练/验证集）
df = stratisfy_df(df, 'strat_fold')
# 保存元数据CSV文件（包含所有ECG的关键信息和标签）
df.to_csv(output_folder + 'cpsc_database.csv', index=False)