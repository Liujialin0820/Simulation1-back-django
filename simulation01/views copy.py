from django.http import JsonResponse
from simulation01.serializers import ParametersModelSerializer, SimDataSerializer
from .models import Parameters, SimulationResult
import numpy as np
import math
from scipy.stats import norm
import numpy_financial as npf
import pandas as pd
from rest_framework.generics import ListCreateAPIView


class ParametersListCreateAPIView(ListCreateAPIView):
    queryset = Parameters.objects.all()
    serializer_class = ParametersModelSerializer


def firstMethod(request):
    # 查询数据库中 name 匹配的第一条记录
    res = Parameters.objects.last()
    serializer = ParametersModelSerializer(instance=res)
    res_data = serializer.data
    print(res_data)
    S = res_data["S0"]
    K = res_data["K"]
    T = res_data["T"]
    r = res_data["r"]
    sigma = res_data["sigma"]
    Y = res_data["Y"]
    μ = res_data["μ"]
    income_tax_rate_I = res_data["income_tax_rate_I"]
    capital_gains_tax_rate_I = res_data["capital_gains_tax_rate_I"]
    capital_gains_tax_rate_G = res_data["capital_gains_tax_rate_G"]
    Franking = res_data["Franking"]
    simulation_step = res_data["simulation_step"]

    def black_scholes_with_dividend(S, K, T, r, sigma, Y, option_type="call"):
        """
        Black-Scholes 公式 (包含分红收益率)

        参数:
        S: 当前股票价格
        K: 期权执行价格
        T: 到期时间 (以年为单位)
        r: 无风险利率 (年化)
        sigma: 波动率 (年化)
        Y: 分红收益率 (年化)
        option_type: "call" 或 "put" (期权类型)

        返回:
        期权价格
        """
        # 计算 d1 和 d2
        d1 = (math.log(S / K) + (r - Y + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
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
    call_price = black_scholes_with_dividend(S, K, T, r, sigma, Y, option_type="call")
    # put_price = black_scholes_with_dividend(S, K, T, r, sigma, Y, option_type="put")

    #  计算分红影响的现值
    def dividend_impact(S, Y, T):
        """
        计算 S(1 - e^(-YT))，即分红影响的现值

        参数:
        S: 当前股票价格
        Y: 分红收益率
        T: 时间 (以年为单位)

        返回:
        分红影响的现值
        """
        return S * (1 - math.exp(-Y * T))

    def Calculate_I(call_price):
        return S / (1 + call_price / dividend_impact(S, Y, T))

    # 根据分红影响的现值计算 I 价格和 G 价格
    I_price = Calculate_I(call_price)  # 计算分红收益产品的价格 (I)
    G_price = S - I_price  # 计算股票价值变化产品的价格 (G)

    #  随机生成股票价格变化
    result = []

    def simulation(n):
        n = int(n)
        for simulation_time in range(n):
            result.append({})
            mean = μ - Y  # 预期回报减去分红收益率
            std_dev = sigma  # 波动率

            # 生成6个随机的股票收益率
            r_values = norm.ppf(np.random.rand(6), loc=mean, scale=std_dev)

            # 半年价格变化
            half_year_price_change = np.exp(r_values * 0.5)

            # 计算未来股票价格，假设初始股票价格为100
            stock_price = 100 * np.cumprod(half_year_price_change)
            result[simulation_time]["stock price"] = np.round(stock_price, 3).tolist()

            #  G: 股票价值变化产品的计算
            G1 = (G_price / S) * min(
                stock_price[-1] - K, 0
            )  # G1 部分：看跌期权的损失部分（基于股票价值变化）
            G2 = max(
                stock_price[-1] - K, 0
            )  # G2 部分：看涨期权的收益部分（基于股票价值变化）
            GCapitalGain = (G1 + G2) * (
                1 - capital_gains_tax_rate_G
            )  # 计算资本增益，扣除公司税（基于股票价值变化）
            G3 = (
                G_price + GCapitalGain
            )  # G的最终价值（包括股票价值变化的收益和税后资本增益）
            G_IRR = 2 * ((G3 / G_price) ** (1 / 6) - 1)  # 计算G部分的IRR（年化收益率）
            result[simulation_time]["G-IRR"] = round(float(G_IRR), 3)
            # I: 股票分红收益产品的计算
            I1 = (I_price / S) * min(
                stock_price[-1] - K, 0
            )  # I1 部分：看跌期权的损失部分（基于分红收益）
            I3 = (
                I_price + (1 - capital_gains_tax_rate_I) * I1
            )  # 计算税后分红收益（基于分红收益）

            # 计算分红影响
            Net_DIV = stock_price * Y / 2  # 计算每期的净分红（考虑分红收益率）
            Gross_DIV = Net_DIV / (1 - 0.3)  # 计算毛分红（假设税费为30%）
            FC = (
                Net_DIV - Gross_DIV
            ) * Franking  # 计算符合税收的分红部分（基于分红收益）
            atax = (Net_DIV - FC) * (
                1 - income_tax_rate_I
            )  # 计算税后分红（基于分红收益）

            # 构建现金流序列（包括初期投资和未来税后分红收益）
            cash_flows = atax
            cash_flows[-1] += I3  # 最后一期现金流加上分红收益的资本增值
            cash_flows = np.insert(
                cash_flows, 0, 0 - I_price
            )  # 初始期的现金流（负的I价格）

            # 计算IRR
            I_irr = npf.irr(cash_flows) * 2  # 计算IRR并乘以2，得到年化IRR
            result[simulation_time]["I-IRR"] = round(I_irr, 3)

        return result

    result = simulation(simulation_step)
    datainstance = SimulationResult.objects.create(data=result)

    # 如果找到数据，返回 jdata，否则返回错误信息
    if res:
        return JsonResponse(
            {
                "result": "created new simulation data",
                "Parameters": res_data,
                "id": datainstance.id,
            }
        )
    else:
        return JsonResponse({"error": "Data not found or invalid format"}, status=404)


def get_IRR_data(request):
    data = SimulationResult.objects.last()
    if not data:
        return JsonResponse({"error": "没有找到任何记录。"}, status=404)

    # 使用序列化器序列化数据
    serializer = SimDataSerializer(instance=data)
    serialized_data = serializer.data  # 正确获取序列化后的数据
    sheet_data = serialized_data.get("data", {})

    try:
        sheet = pd.DataFrame(sheet_data)
        df = sheet[["G-IRR", "I-IRR"]]
        # 定义区间范围和标签
        step = 0.01
        bins = np.arange(-0.2, 0.35 + step, step)  # 从-1到1，步长0.01

        # 生成标签，例如 "-1.00--0.99", "-0.99--0.98", ..., "0.99-1.00"
        labels = [f"{bins[i+1]*100:.0f}%" for i in range(len(bins) - 1)]

        # 分桶
        # 分别对每列进行分桶
        df["G-IRR_bins"] = pd.cut(df["G-IRR"], bins=bins, labels=labels, right=False)
        df["I-IRR_bins"] = pd.cut(df["I-IRR"], bins=bins, labels=labels, right=False)

        # 分别统计每列的频数
        g_irr_stats = df["G-IRR_bins"].value_counts().sort_index().to_dict()
        i_irr_stats = df["I-IRR_bins"].value_counts().sort_index().to_dict()
    except Exception as e:
        return JsonResponse({"error": f"创建 DataFrame 时出错: {str(e)}"}, status=500)

    return JsonResponse(
        {
            "charData": [
                {"name": "g_irr_stats_values", "data": list(g_irr_stats.values())},
                {"name": "i_irr_stats_values", "data": list(i_irr_stats.values())},
            ],
            "xAxisData": list(g_irr_stats.keys()),
        },
        status=200,
        safe=False,
    )
