from rest_framework import serializers
from .models import Parameters, SimulationResult


class SimDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = SimulationResult
        fields = "__all__"


class ParametersModelSerializer(serializers.ModelSerializer):
    help_texts = serializers.SerializerMethodField()

    class Meta:
        model = Parameters
        exclude = ["user", "name", "id"]

    def get_help_texts(self, obj):
        """
        Retrieves the help_texts for each field in the model.
        """
        return {
            field.name: field.help_text
            for field in obj._meta.get_fields()
            if hasattr(field, "help_text") and field.help_text
        }
