

def parabolic_equation(data, a, b, c, d, e):
    """
    Equation for a paraboloid to fit to the cost function for determining the index of refraction
    of a material
    :param data:
    :param a:
    :param b:
    :param c:
    :param d:
    :param e:
    :return:
    """
    y, x = data

    paraboloid = ((x - a) / b) ** 2 + ((y - d) / e) ** 2 + c

    return paraboloid
