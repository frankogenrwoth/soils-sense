from django import template

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

@register.filter(name="is_inbuilt")
def is_inbuilt(model_type, algorithm):
    """
    Check if the model is inbuilt. i.e. has version numbers to indicate it is a version of an inbuilt model.
     - inbuilt models don't have a version number.
    Usage: {{ model_type|isInbuilt:algorithm }}
    """
    if model_type.find("version") != -1:
        return True
    else:
        return False

@register.filter(name="clean_ml_name")
def clean_ml_name(value):
    """
    Clean the model name to remove underscores and capitalize the first letter.
    Usage: {{ model_type|clean_ml_name }}
    """
    return value.replace("_", " ").title()