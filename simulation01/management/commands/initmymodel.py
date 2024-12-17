from django.core.management.base import BaseCommand
from simulation01.models import JModel


class Command(BaseCommand):
    def handle(self, *args, **options):
        JModel.objects.create(
            jdata={
                "xdata": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                "ydata": [120, 200, 150, 80, 70, 110, 130],
            },
            jname="BasicBar",
        )

        self.stdout.write("数据初始化成功！")
