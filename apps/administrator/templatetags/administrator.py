from django import template
from datetime import datetime

register = template.Library()


@register.filter(name="zip")
def zip_lists(a, b):
    """
    Zip two lists together for parallel iteration in templates.
    Usage: {% for x, y in list1|zip:list2 %}
    """
    return zip(a, b)


@register.filter(name="get_dict_item")
def get_dict_item(dictionary, key):
    """
    Get dictionary value by key, handling special characters in key names.
    Usage: {{ dict|get_dict_item:'25%' }}
    """
    try:
        return dictionary.get(str(key))
    except:
        return None


@register.filter(name="percentage")
def percentage(value):
    """
    Format a number as a percentage.
    Usage: {{ value|percentage }}
    """
    try:
        return f"{float(value):.2f}%"
    except (ValueError, TypeError):
        return value


@register.filter(name="round_number")
def round_number(value, decimal_places=2):
    """
    Round a number to specified decimal places.
    Usage: {{ value|round_number:2 }}
    """
    try:
        return round(float(value), decimal_places)
    except (ValueError, TypeError):
        return value


@register.filter(name="clean_ml_name")
def clean_ml_name(value):
    """
    Clean the model name to remove underscores and capitalize the first letter.
    Usage: {{ model_type|clean_ml_name }}
    """
    return value.replace("_", " ").title()


@register.filter(name="is_inbuilt")
def is_inbuilt(value):
    """
    Check if the model is inbuilt. i.e. has version numbers to indicate it is a version of an inbuilt model.
     - inbuilt models have a version number.
    Usage: {{ model_type|is_inbuilt }}
    """
    return str(value).find("v0.") != -1


@register.filter(name="strftime")
def strftime(value, format="%H:%M:%S"):
    """
    Format a time string to a given format. for example 2 seconds or 2.5 seconds or 2.5 minutes or 2.5 hours.
    Usage: {{ value|strftime:"%H:%M:%S" }}
    """
    if value < 60:
        return f"{value:.2f} s"
    elif value < 3600:
        return f"{value/60:.2f} min"
    elif value < 86400:
        return f"{value/3600:.2f} hr"
    else:
        return f"{value/86400:.2f} d"


@register.filter(name="round")
def round_number(value, decimal_places=4):
    """
    Round a number to specified decimal places.
    Usage: {{ value|round:2 }}
    """
    return round(float(value), decimal_places)


@register.filter
def get_item(dictionary, key):
    """Get an item from a dictionary using bracket notation in templates."""
    return dictionary.get(key)


@register.filter
def count_active_sensors(sensor_stats):
    """Count the number of active sensors in sensor_stats dictionary."""
    return sum(1 for stats in sensor_stats.values() if stats.get("status") == "Active")


@register.filter
def sum_readings(sensor_stats):
    """Sum up the total readings from all sensors."""
    return sum(stats.get("reading_count", 0) for stats in sensor_stats.values())


@register.filter
def average_moisture(sensor_stats):
    """Calculate the average moisture across all sensors."""
    moisture_values = [
        stats.get("avg_moisture")
        for stats in sensor_stats.values()
        if stats.get("avg_moisture") is not None
    ]
    if not moisture_values:
        return None
    return round(sum(moisture_values) / len(moisture_values), 1)
