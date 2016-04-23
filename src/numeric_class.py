classes_to_numbers = {'d ': 0, 'h ': 1, 's ': 2, 'o ': 3}
numbers_to_classes = dict((v, k) for k, v in classes_to_numbers.iteritems())


def get_numeric_class(row):
    if row['class'] == 'd ':
        return classes_to_numbers['d ']
    if row['class'] == 'h ':
        return classes_to_numbers['h ']
    if row['class'] == 's ':
        return classes_to_numbers['s ']
    if row['class'] == 'o ':
        return classes_to_numbers['o ']
    return -111111
