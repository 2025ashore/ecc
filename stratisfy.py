from sklearn.model_selection import StratifiedKFold
import pandas as pd


def stratisfy_df(df, fold_col, n_splits=10, random_state=42):
    """
    为DataFrame添加分层折叠列（模拟stratisfy_df功能）
    df: 输入DataFrame
    fold_col: 折叠列的名称（如'strat_fold'）
    n_splits: 折叠数量（默认10折）
    """
    # 假设按某个标签列分层（需根据实际数据调整，例如多标签可合并为字符串）
    # 这里以'scp_codes'的哈希值作为分层依据（示例）
    df['strat_key'] = df['scp_codes'].apply(lambda x: hash(tuple(sorted(x.keys()))))

    # 初始化分层K折
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # 为每行分配折叠编号
    df[fold_col] = 0
    for fold, (_, test_idx) in enumerate(skf.split(df, df['strat_key'])):
        df.loc[test_idx, fold_col] = fold + 1  # 折叠编号从1开始（与原代码保持一致）

    df.drop(columns=['strat_key'], inplace=True)
    return df