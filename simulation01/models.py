from django.db import models
from django.core.validators import MaxValueValidator


# Create your models here.
class Parameters(models.Model):
    name = models.CharField(max_length=256)
    user = models.CharField(max_length=256, blank=True)
    S0 = models.FloatField(default=100, help_text="Stock price at time 0")
    K = models.FloatField(default=100, help_text="Strike/Exercise Price")
    T = models.FloatField(default=3, help_text="Expiration Time")
    r = models.FloatField(default=0.045, help_text="Risk-free interest rate (annual)")
    sigma = models.FloatField(default=0.15, help_text="Volatility (annual)")
    Y = models.FloatField(default=0.035, help_text="Dividend yield (annual)")
    μ = models.FloatField(default=0.095, help_text="Expected total return")
    C = models.FloatField(
        default=1,
        help_text="The price of call option",
    )
    I0 = models.FloatField(default=0, help_text="Price of I product")
    G0 = models.FloatField(default=0, help_text="Price of G product")
    Franking = models.FloatField(
        default=0.9, help_text="Franking credit rate (assumed 90%)"
    )
    simulation_step = models.IntegerField(
        default=10,
        help_text="Number of simulation steps",
        validators=[MaxValueValidator(1000000)],
    )
    Growth_Party_Income_tax = models.FloatField(
        default=0.30, help_text="Income tax rate for Growth Party"
    )
    Growth_Party_Cap_gains_tax = models.FloatField(
        default=0.235, help_text="Capital gains tax rate for Growth Party"
    )
    Income_Party_Income_tax = models.FloatField(
        default=0.30, help_text="Income tax rate for Income Party"
    )
    Income_Party_Cap_gains_tax = models.FloatField(
        default=0.235, help_text="Capital gains tax rate for Income Party"
    )


class SimulationResult(models.Model):
    data = models.JSONField()
    static = models.JSONField()
    tax = models.JSONField()
    tax_static = models.JSONField()
