from datetime import datetime


def convert_dates_in_db_filter(db_filter, is_period=False):
    if isinstance(db_filter, list):
        result = []
        for el in db_filter:
            result.append(convert_dates_in_db_filter(el, is_period))
    elif isinstance(db_filter, dict):
        result = {}
        for k, v in db_filter.items():
            if k == "period":
                result[k] = convert_dates_in_db_filter(v, True)
            else:
                result[k] = convert_dates_in_db_filter(v, is_period)
    elif isinstance(db_filter, str) and is_period:
        result = datetime.strptime(db_filter, "%d.%m.%Y")
    else:
        result = db_filter

    return result
