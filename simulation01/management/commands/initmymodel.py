from django.core.management.base import BaseCommand
from simulation01.models import Parameters


class Command(BaseCommand):
    def handle(self, *args, **options):
        Parameters.objects.create(
            data={
                "S0": {"value": 100, "desc": "Stock price at time 0"},
                "ST": {"value": 100, "desc": "Stock price at time T"},
                "K": {"value": 100, "desc": "Strike/Exercise Price"},
                "T": {"value": 3, "desc": "Expiration Time"},
                "r": {"value": 0.045, "desc": "Risk-free interest rate (annual)"},
                "sigma": {"value": 0.15, "desc": "Volatility (annual)"},
                "Y": {"value": 0.035, "desc": "Dividend yield (annual)"},
                "μ": {"value": 0.095, "desc": "Expected total return"},
                "income_tax_rate_I": {
                    "value": 0.150,
                    "desc": "Income tax rate (I product)",
                },
                "capital_gains_tax_rate_I": {
                    "value": 0.100,
                    "desc": "Capital gains tax rate (I product)",
                },
                "capital_gains_tax_rate_G": {
                    "value": 0.235,
                    "desc": "Capital gains tax rate (G product)",
                },
                "Franking": {
                    "value": 0.9,
                    "desc": "Franking credit rate (assumed 90%)",
                },
                "simulation_step": {"value": 10, "desc": "Number of simulation steps"},
            },
            name="Black Scholes",
            user="admin",
        )

        self.stdout.write("success!")
