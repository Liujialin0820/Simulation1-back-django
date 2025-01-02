from django.http import JsonResponse
from simulation01.serializers import ParametersModelSerializer, SimDataSerializer
from .models import Parameters, SimulationResult
import numpy as np
import math
from scipy.stats import norm
import numpy_financial as npf
import pandas as pd
from rest_framework.generics import ListCreateAPIView
import json


class ParametersListCreateAPIView(ListCreateAPIView):
    queryset = Parameters.objects.all()
    serializer_class = ParametersModelSerializer


def firstMethod(request):
    # 查询数据库中 name 匹配的第一条记录
    model = request.GET.get("model")
    res = Parameters.objects.last()
    serializer = ParametersModelSerializer(instance=res)
    res_data = serializer.data
    S = res_data["S0"]
    K = res_data["K"]
    T = res_data["T"]
    r = res_data["r"]
    sigma = res_data["sigma"]
    Y = res_data["Y"]
    μ = res_data["μ"]
    Franking = res_data["Franking"]
    n = res_data["simulation_step"]
    Family_Office_Income_tax = res_data["Family_Office_Income_tax"]
    Family_Office_Cap_gains_tax = res_data["Family_Office_Cap_gains_tax"]
    Super_Fund_Income_tax = res_data["Super_Fund_Income_tax"]
    Super_Fund_Cap_gains_tax = res_data["Super_Fund_Cap_gains_tax"]

    if model == "black":

        def black_scholes_with_dividend(S, K, T, r, sigma, Y, option_type="call"):
            # 计算 d1 和 d2
            d1 = (math.log(S / K) + (r - Y + 0.5 * sigma**2) * T) / (
                sigma * math.sqrt(T)
            )
            d2 = d1 - sigma * math.sqrt(T)

            # 看涨期权（Call Option）定价
            if option_type == "call":
                price = S * math.exp(-Y * T) * norm.cdf(d1) - K * math.exp(
                    -r * T
                ) * norm.cdf(d2)
            # 看跌期权（Put Option）定价
            elif option_type == "put":
                price = K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(
                    -Y * T
                ) * norm.cdf(-d1)
            else:
                raise ValueError("option_type 必须是 'call' 或 'put'")

            return price

        # 计算看涨期权和看跌期权价格
        call_price = black_scholes_with_dividend(
            S, K, T, r, sigma, Y, option_type="call"
        )

        tmp = S - S * math.exp(-Y * T)
        tmp = call_price / tmp
        I0 = S / (1 + tmp)
        G0 = S - I0
    if model == "customise":
        I0 = res_data["I0"]
        G0 = res_data["G0"]
        call_price = res_data["C"]

    sheet = pd.DataFrame(index=range(n))
    sheet["I0"] = I0
    sheet["G0"] = G0
    res.I0 = round(I0, 2)
    res.G0 = round(G0, 2)
    res.C = round(call_price, 2)

    # r
    mean = μ - Y  # 预期回报减去分红收益率
    std_dev = sigma  # 波动率
    r_values = np.random.normal(loc=mean, scale=std_dev, size=(n, 6))
    exp_r_values = np.exp(r_values * 0.5)
    columns = [f"exp_r{i}" for i in range(1, 7)]
    sheet[columns] = exp_r_values

    # S
    scolumn = []
    for i in range(1, 7):
        columnName = f"S{i*0.5}"
        scolumn.append(columnName)
        rColumn = f"exp_r{i}"
        if i == 1:
            sheet[columnName] = sheet[rColumn] * S
        else:
            columnNamePre = f"S{(i-1)*0.5}"
            sheet[columnName] = sheet[rColumn] * sheet[columnNamePre]
    S_values = np.array(sheet[scolumn])

    # G3
    sheet["tmp1"] = (G0 / S) * np.minimum(sheet["S3.0"] - K, 0)
    sheet["tmp2"] = np.maximum(sheet["S3.0"] - K, 0)
    sheet["Capital_gain_Family_Office"] = (sheet["tmp1"] + sheet["tmp2"]) * (
        1 - Family_Office_Cap_gains_tax
    )
    sheet["G3"] = sheet["G0"] + sheet["Capital_gain_Family_Office"]

    sheet["Product_Family_Office_after_tax_income_G_irr_values"] = 2 * (
        (sheet["G3"] / sheet["G0"]) ** (1 / 6) - 1
    )

    sheet["tmp3"] = np.minimum(sheet["S3.0"] - K, 0) * (I0 / S)
    sheet["I3"] = sheet["tmp3"] * (1 - Super_Fund_Cap_gains_tax) + I0

    Div_values = S_values * Y * 0.5
    columns = [f"Div{i}" for i in range(1, 7)]
    sheet[columns] = Div_values

    Gross_Div_values = Div_values / (1 - 0.3)
    columns = [f"Gross_Div{i}" for i in range(1, 7)]
    sheet[columns] = Gross_Div_values

    FC_vales = (Gross_Div_values - Div_values) * Franking
    columns = [f"FC{i}" for i in range(1, 7)]
    sheet[columns] = FC_vales

    Share_Fammily_Office_after_tax_income = (FC_vales + Div_values) * (
        1 - Family_Office_Income_tax
    )
    Share_Fammily_Office_after_tax_income = np.insert(
        Share_Fammily_Office_after_tax_income, 0, -S, axis=1
    )
    Share_Fammily_Office_after_tax_income[:, 6] += (sheet["S3.0"] - S) * (
        1 - Family_Office_Cap_gains_tax
    ) + S
    sheet["Share_Fammily_Office_after_tax_income_irr_values"] = (
        np.apply_along_axis(npf.irr, axis=1, arr=Share_Fammily_Office_after_tax_income)
        * 2
    )

    Product_Super_Fund_after_tax_income = (FC_vales + Div_values) * (
        1 - Super_Fund_Income_tax
    )
    Product_Super_Fund_after_tax_income = np.insert(
        Product_Super_Fund_after_tax_income, 0, -I0, axis=1
    )
    Product_Super_Fund_after_tax_income[:, 6] += sheet["I3"]

    sheet["Product_Super_Fund_after_tax_income_I_irr_values"] = (
        np.apply_along_axis(npf.irr, axis=1, arr=Product_Super_Fund_after_tax_income)
        * 2
    )
    print(Product_Super_Fund_after_tax_income[1])

    Share_Super_Fund_after_tax_income = (FC_vales + Div_values) * (
        1 - Super_Fund_Income_tax
    )
    Share_Super_Fund_after_tax_income = np.insert(
        Share_Super_Fund_after_tax_income, 0, -S, axis=1
    )
    Share_Super_Fund_after_tax_income[:, 6] += S + (sheet["S3.0"] - S) * (
        1 - Super_Fund_Cap_gains_tax
    )
    sheet["Share_Super_Fund_after_tax_income_irr_values"] = (
        np.apply_along_axis(npf.irr, axis=1, arr=Share_Super_Fund_after_tax_income) * 2
    )

    # 两个相加
    U = Product_Super_Fund_after_tax_income
    U[:, 0] -= G0
    U[:, 6] += sheet["G3"]
    columns = [f"Product Sum Cashflow{i}" for i in range(0, 7)]
    sheet[columns] = U
    sheet["Sum_of_the_Growth_and_Income_products_irr"] = (
        np.apply_along_axis(npf.irr, axis=1, arr=U) * 2
    )

    sheet_result = sheet[
        [
            "Share_Super_Fund_after_tax_income_irr_values",
            "Product_Super_Fund_after_tax_income_I_irr_values",
            "Share_Fammily_Office_after_tax_income_irr_values",
            "Product_Family_Office_after_tax_income_G_irr_values",
            "Sum_of_the_Growth_and_Income_products_irr",
        ]
    ]
    renamed_columns = {
        "Share_Super_Fund_after_tax_income_irr_values": "SuperFund_Share_IRR",
        "Product_Super_Fund_after_tax_income_I_irr_values": "SuperFund_Product_IRR",
        "Share_Fammily_Office_after_tax_income_irr_values": "FamilyOffice_Share_IRR",
        "Product_Family_Office_after_tax_income_G_irr_values": "FamilyOffice_Product_IRR",
        "Sum_of_the_Growth_and_Income_products_irr": "Product_Sum_IRR",
    }
    sheet_result = sheet_result.rename(columns=renamed_columns)

    # 计算均值和标准差
    mean_std_result = sheet_result.agg(["mean", "std"])
    mean_std_result.round(4)

    SimulationResult.objects.create(
        data=sheet_result.to_json(), static=mean_std_result.round(4).to_json()
    )
    sheet.to_csv("simulation_process.csv")
    res.save()
    return JsonResponse(
        {
            "result": "created new simulation data",
        }
    )


def get_IRR_data(request):
    data = SimulationResult.objects.last()
    if not data:
        return JsonResponse({"error": "没有找到任何记录。"}, status=404)

    # 使用序列化器序列化数据
    serializer = SimDataSerializer(instance=data)
    serialized_data = serializer.data  # 正确获取序列化后的数据
    sheet_data = serialized_data.get("data", {})
    sheet_data = json.loads(sheet_data)
    static_data = serialized_data.get("static", {})

    try:
        sheet = pd.DataFrame(sheet_data)
        df = pd.DataFrame()
        # 定义区间范围和标签
        step = 0.01
        bins = np.arange(-0.2, 0.35 + step, step)  # 从-1到1，步长0.01

        # 生成标签，例如 "-1.00--0.99", "-0.99--0.98", ..., "0.99-1.00"
        labels = [f"{bins[i+1]*100:.0f}%" for i in range(len(bins) - 1)]

        # 分桶
        charData = []
        # 分别对每列进行分桶
        for key in sheet.keys():
            df[key] = pd.cut(sheet[key], bins=bins, labels=labels, right=False)
            charData.append(
                {
                    "name": key,
                    "data": list(
                        df[key].value_counts().sort_index().to_dict().values()
                    ),
                }
            )

        # 分别统计每列的频数
    except Exception as e:
        return JsonResponse({"error": f"创建 DataFrame 时出错: {str(e)}"}, status=500)

    return JsonResponse(
        {
            "charData": charData,
            "xAxisData": list(
                df[sheet.keys()[0]].value_counts().sort_index().to_dict().keys()
            ),
            "static_data": static_data,
        },
        status=200,
        safe=False,
    )
