from django.core.management.base import BaseCommand
from simulation01.models import Parameters


class Command(BaseCommand):
    def handle(self, *args, **options):
        Parameters.objects.create(
            data={
                "S0": {"value": 0, "desc": "Stock price at time 0"},
                "ST": {"value": 100, "desc": "stock price at time T"},
                "K": {"value": 45, "desc": "Strike/Exercise Price"},
                "r": {"value": 0.05},
                "σ": {"value": 0.20},
                "Y": {"value": 0.04},
                "T": {"value": 0.5, "desc": "Yield"},
            },
            name="Black Scholes",
        )
        Parameters.objects.create(
            data={
                "S0": {"value": 40, "desc": "Stock price at time 0"},
                "ST": {"value": 100, "desc": "stock price at time T"},
                "K": {"value": 45, "desc": "Strike/Exercise Price"},
                "r": {"value": 0.05},
                "σ": {"value": 0.20},
                "T": {"value": 0.5, "desc": "Yield"},
                "Y": {"value": 0.04},
            },
            name="Call Option",
        )
        Parameters.objects.create(
            data={
                "S0": {"value": 40, "desc": "Stock price at time 0"},
                "ST": {"value": 100, "desc": "stock price at time T"},
                "K": {"value": 45, "desc": "Strike/Exercise Price"},
                "r": {"value": 0.05},
                "σ": {"value": 0.20},
                "T": {"value": 0.5, "desc": "Yield"},
                "Y": {"value": 0.04},
            },
            name="Put Option",
        )

        self.stdout.write("success!")
