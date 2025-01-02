# management/commands/init_parameters.py

from django.core.management.base import BaseCommand
from simulation01.models import Parameters


class Command(BaseCommand):
    help = "Initialize a Parameters entry in the database with default values."

    def handle(self, *args, **options):
        Parameters.objects.create(
            name="Black Scholes",
            user="admin",
            S0=100,
            K=100,
            T=3,
            r=0.045,
            sigma=0.15,
            Y=0.035,
            μ=0.095,
            Franking=0.9,
            Family_Office_Income_tax=0.30,
            Family_Office_Cap_gains_tax=0.235,
            Super_Fund_Income_tax=0.30,
            Super_Fund_Cap_gains_tax=0.235,
            simulation_step=10,
        )
        self.stdout.write(self.style.SUCCESS("Parameters entry successfully created!"))
