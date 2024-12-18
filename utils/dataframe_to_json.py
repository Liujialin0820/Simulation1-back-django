import pandas as pd
import json


def dataframe_to_json(sheet):
    """
    将 DataFrame 转换为一个 JSON 格式，包含 'categories' 和 'values'。

    参数：
    df : pandas.DataFrame
        输入的 DataFrame。
    category_column : str
        用作 'categories' 的列名。
    value_column : str
        用作 'values' 的列名。

    返回：
    dict
        包含 'categories' 和 'values' 的字典。
    """
    result = {}
    for col in sheet.keys().tolist():
        result[col] = sheet[col].tolist()
    return result
