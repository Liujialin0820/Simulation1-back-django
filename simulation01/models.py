from django.db import models
from django.core.validators import MaxValueValidator


# Create your models here.
class Parameters(models.Model):
    name = models.CharField(max_length=256)
    user = models.CharField(max_length=256, blank=True)
    S0 = models.FloatField(default=100, help_text="Stock price at time 0")
    ST = models.FloatField(default=100, help_text="Stock price at time T")
    K = models.FloatField(default=100, help_text="Strike/Exercise Price")
    T = models.FloatField(default=3, help_text="Expiration Time")
    r = models.FloatField(default=0.045, help_text="Risk-free interest rate (annual)")
    sigma = models.FloatField(default=0.15, help_text="Volatility (annual)")
    Y = models.FloatField(default=0.035, help_text="Dividend yield (annual)")
    μ = models.FloatField(default=0.095, help_text="Expected total return")
    Franking = models.FloatField(
        default=0.9, help_text="Franking credit rate (assumed 90%)"
    )
    simulation_step = models.IntegerField(
        default=10,
        help_text="Number of simulation steps",
        validators=[MaxValueValidator(1000000)],
    )
    Family_Office_Income_tax = models.FloatField(
        default=0.30, help_text="Income tax rate for Family Office"
    )
    Family_Office_Cap_gains_tax = models.FloatField(
        default=0.235, help_text="Capital gains tax rate for Family Office"
    )
    Super_Fund_Income_tax = models.FloatField(
        default=0.30, help_text="Income tax rate for Super Fund"
    )
    Super_Fund_Cap_gains_tax = models.FloatField(
        default=0.235, help_text="Capital gains tax rate for Super Fund"
    )


class SimulationResult(models.Model):
    data = models.JSONField()
    static = models.JSONField()
