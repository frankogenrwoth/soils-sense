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
