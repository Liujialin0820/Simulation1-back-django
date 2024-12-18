from django.http import JsonResponse
from .models import Parameters
import pandas as pd
import numpy as np
from utils.dataframe_to_json import dataframe_to_json
from django.shortcuts import get_object_or_404
import json


def firstMethod(request):
    # 从请求中获取 'name' 参数
    name = request.GET.get("name")

    # 查询数据库中 name 匹配的第一条记录
    res = Parameters.objects.filter(name=name).first()
    res_data = res.data
    sheet = pd.DataFrame()
    data = np.arange(0, 101, 5)
    sheet["x"] = data
    sheet["y"] = np.where(
        sheet["x"] < res_data["K"]["value"], 0, sheet["x"] - res_data["K"]["value"]
    )
    result = dataframe_to_json(sheet)
    # 如果找到数据，返回 jdata，否则返回错误信息
    if res and isinstance(res.data, dict):
        print(result)
        return JsonResponse({"data": result, "K": res_data["K"], "id": res.id})
    else:
        return JsonResponse({"error": "Data not found or invalid format"}, status=404)


def update_parameter(request, pk):
    """
    使用 PUT 方法更新指定 Parameters 对象的 data 字段中的 JSON 值
    """
    if request.method == "PUT":
        try:
            # 获取指定的 Parameters 对象
            parameter = get_object_or_404(Parameters, pk=pk)

            # 解析请求体中的 JSON 数据
            body = json.loads(request.body)

            # 验证请求数据中是否包含 `key` 和 `value`
            key = body.get("key")
            value = body.get("value")
            if key is None or value is None:
                return JsonResponse(
                    {"error": "Missing 'key' or 'value' in request body"}, status=400
                )

            # 更新 data 字段的 JSON 数据
            data = parameter.data or {}
            data[key]["value"] = float(value)
            parameter.data = data
            parameter.save()

            return JsonResponse(
                {"message": "Parameter updated successfully", "data": parameter.data},
                status=200,
            )

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON in request body"}, status=400)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method"}, status=405)
