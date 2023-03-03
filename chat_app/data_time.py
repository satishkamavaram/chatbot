import datetime
import calendar

data = {'year': ['2023', '2023'], 'time': ['2', '1', '3', '2']}
data = {'month': ['january', 'mar'], 'year': ['2021', '2023']}
data = {'month': ['february', 'march']}
data = {'time': ['1', '28'], 'month': ['jan', 'sep'], 'year': ['2022', '2023']}
data = {'state': ['today']}
data = {'state': ['yesterday']}
data = {'month': ['apr']}
data = {'time': ['09'], 'month': ['dec'], 'year': ['2023']}
data = {'state': ['last'], 'time': ['1'], 'unit': ['month']}
data = {'state': ['last'], 'time': ['3'], 'unit': ['yr']}
data = {'state': ['last'], 'unit': ['week']}
data = {'state': ['last'], 'time': ['3']}
data = {'state': ['last'], 'time': ['3'], 'unit': ['yr']}
data = {'state': ['last'], 'unit': ['week']}
data = {'state': ['last'], 'time': ['3']}
data = {'state': ['current'], 'unit': ['yr']}
data = {'state': ['current']}

# print(data.get('year'))
month_dict = {
    "jan": '01', "feb": '02', "mar": '03', "apr": '04', "may": '05', "jun": '06',
    "jul": '07', "aug": '08', "sep": '09', "oct": '10', "nov": '11', "dec": '12',
    "january": '01', "february": '02', "march": '03', "april": '04', "may": '05', "june": '06',
    "july": '07', "august": '08', "september": '09', "october": '10', "november": '11', "december": '12'
}

time_dict = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
             "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
             "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
             "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "1st": 1,
             "2nd": 2, "3rd": 3, "4th": 4, "5th": 5, "6th": 6, "7th": 7, "8th": 8, "9th": 9, "10th": 10, "11th": 11,
             "12th": 12, "13th": 13, "14th": 14, "15th": 15, "16th": 16, "17th": 17, "18th": 18, "19th": 19, "20th": 20,
             "21st": 21, "22nd": 22, "23rd": 23,
             "24th": 24, "25th": 25, "26th": 26, "27th": 27, "28th": 28, "29th": 29, "30th": 30, "31st": 31}


def get_month_status(month):
    today = datetime.date.today()
    if today.month > month:
        return "completed"
    elif today.month == month:
        if today.day > 28:
            if month == 2 and (today.year % 4 == 0 and (today.year % 100 != 0 or today.year % 400 == 0)):
                return "running" if today.day <= 29 else "completed"
            else:
                return "running" if today.day <= 28 else "completed"
        else:
            return "running"
    else:
        return "upcoming"


def subtract_months(date, months):
    """
    Subtract the given number of months from the given date.
    """
    year = date.year
    month = date.month - months
    if month <= 0:
        year -= 1
        month += 12
    day = min(date.day, (date.replace(year=year, month=month, day=1) - datetime.timedelta(days=1)).day)
    return date.replace(year=year, month=month, day=day)


def get_start_end_date(data: dict, data_format: str = '%Y-%m-%dT%H:%M:%S.%f%z'):
    if 'time' in data:
        for index, _ in enumerate(data['time']):
            if data['time'][index] in time_dict:
                data['time'][index] = time_dict.get(data['time'][index])

    print(data)
    if 'year' in data and len(data['year']) == 2 and 'month' in data and len(
            data['month']) == 2 and 'time' in data and len(data['time']) == 2:
        st_dt = datetime.datetime(int(data['year'][0]), int(month_dict[data['month'][0]]), int(data['time'][0]), 0, 0,
                                  0).astimezone()
        ed_dt = datetime.datetime(int(data['year'][1]), int(month_dict[data['month'][1]]), int(data['time'][1]), 23, 59,
                                  59).astimezone()
        start_date = st_dt.strftime(data_format)
        end_date = ed_dt.strftime(data_format)
        print(start_date)
        print(end_date)
        return start_date, end_date
    elif 'year' in data and len(data['year']) == 2 and 'time' in data and len(data['time']) == 4:
        st_dt = datetime.datetime(int(data['year'][0]), int(data['time'][0]), int(data['time'][1]), 0, 0,
                                  0).astimezone()
        ed_dt = datetime.datetime(int(data['year'][1]), int(data['time'][2]), int(data['time'][3]), 23, 59,
                                  59).astimezone()
        start_date = st_dt.strftime(data_format)
        end_date = ed_dt.strftime(data_format)
        print(start_date)
        print(end_date)
        return start_date, end_date
    elif 'year' in data and len(data['year']) == 2 and 'month' in data and len(data['month']) == 2:
        last_day = calendar.monthrange(int(data['year'][1]), int(month_dict[data['month'][1]]))[1]
        st_dt = datetime.datetime(int(data['year'][0]), int(month_dict[data['month'][0]]), 1, 0, 0, 0).astimezone()
        ed_dt = datetime.datetime(int(data['year'][1]), int(month_dict[data['month'][1]]), last_day, 23, 59,
                                  59).astimezone()
        start_date = st_dt.strftime(data_format)
        end_date = ed_dt.strftime(data_format)
        print(start_date)
        print(end_date)
        return start_date, end_date
    elif 'month' in data and len(data['month']) == 2:
        current_year = datetime.date.today().year
        # print(current_year)
        start_month = int(month_dict[data['month'][0]])
        start_month_status = get_month_status(start_month)
        if start_month_status == "completed":
            start_year = current_year
        elif start_month_status == "running":
            start_year = current_year
        else:
            start_year = current_year - 1

        end_month = int(month_dict[data['month'][1]])
        end_month_status = get_month_status(end_month)
        # print(start_month_status)
        # print(end_month_status)
        if end_month_status == "completed":
            if start_month > end_month and start_month_status == "completed":
                end_year = current_year + 1
            else:
                end_year = current_year
            last_day = calendar.monthrange(end_year, end_month)[1]
        elif end_month_status == "running":
            end_year = current_year
            last_day = datetime.date.today().day
        else:
            if start_month <= end_month and start_month_status == 'upcoming':
                end_year = current_year - 1
            else:
                end_year = current_year
            last_day = calendar.monthrange(end_year, end_month)[1]
        # print(last_day)
        st_dt = datetime.datetime(start_year, start_month, 1, 0, 0, 0).astimezone()
        ed_dt = datetime.datetime(end_year, end_month, last_day, 23, 59, 59).astimezone()
        start_date = st_dt.strftime(data_format)
        end_date = ed_dt.strftime(data_format)
        print(start_date)
        print(end_date)
        return start_date, end_date
    elif 'state' in data and len(data['state']) == 1 and 'today' in data['state']:
        year = datetime.date.today().year
        month = datetime.date.today().month
        day = datetime.date.today().day
        st_dt = datetime.datetime(year, month, day, 0, 0, 0).astimezone()
        ed_dt = datetime.datetime(year, month, day, 23, 59, 59).astimezone()
        start_date = st_dt.strftime(data_format)
        end_date = ed_dt.strftime(data_format)
        print(start_date)
        print(end_date)
        return start_date, end_date
    elif 'state' in data and len(data['state']) == 1 and 'yesterday' in data['state']:
        yesterday = datetime.date.today() - datetime.timedelta(days=1)
        year = yesterday.year
        month = yesterday.month
        day = yesterday.day
        st_dt = datetime.datetime(year, month, day, 0, 0, 0).astimezone()
        ed_dt = datetime.datetime(year, month, day, 23, 59, 59).astimezone()
        start_date = st_dt.strftime(data_format)
        end_date = ed_dt.strftime(data_format)
        print(start_date)
        print(end_date)
        return start_date, end_date
    elif 'year' in data and len(data['year']) == 1 and 'month' in data and len(
            data['month']) == 1 and 'time' in data and len(data['time']) == 1:
        st_dt = datetime.datetime(int(data['year'][0]), int(month_dict[data['month'][0]]), int(data['time'][0]), 0, 0,
                                  0).astimezone()
        ed_dt = datetime.datetime(int(data['year'][0]), int(month_dict[data['month'][0]]), int(data['time'][0]), 23, 59,
                                  59).astimezone()
        start_date = st_dt.strftime(data_format)
        end_date = ed_dt.strftime(data_format)
        print(start_date)
        print(end_date)
        return start_date, end_date
    elif 'month' in data and len(data['month']) == 1:
        current_year = datetime.date.today().year
        start_month = int(month_dict[data['month'][0]])
        start_month_status = get_month_status(start_month)
        if start_month_status == "completed":
            start_year = current_year
            last_day = calendar.monthrange(start_year, start_month)[1]
        elif start_month_status == "running":
            start_year = current_year
            last_day = datetime.date.today().day
        else:
            start_year = current_year - 1
            last_day = calendar.monthrange(start_year, start_month)[1]
        st_dt = datetime.datetime(start_year, start_month, 1, 0, 0, 0).astimezone()
        ed_dt = datetime.datetime(start_year, start_month, last_day, 23, 59, 59).astimezone()
        start_date = st_dt.strftime(data_format)
        end_date = ed_dt.strftime(data_format)
        print(start_date)
        print(end_date)
        return start_date, end_date
    elif 'state' in data and len(data['state']) == 1:
        if 'time' not in data:
            time = 1
        elif len(data['time']) == 1:
            time = data['time'][0]
        if 'unit' not in data:
            unit = 'month'
        elif len(data['unit']) == 1:
            unit = data['unit'][0]
        print(time, unit, data['state'])
        if data['state'][0] == 'last':
            if unit == 'hr' or unit == 'hour' or unit == 'hours' or unit == 'hrs':
                now = datetime.datetime.now()
                last = now - datetime.timedelta(hours=int(time))
            elif unit == 'day' or unit == 'days':
                now = datetime.datetime.now()
                last = now - datetime.timedelta(days=int(time))
            elif unit == 'week' or unit == 'weeks':
                now = datetime.datetime.now()
                last = now - datetime.timedelta(weeks=int(time))
            elif unit == 'month' or unit == 'months':
                now = datetime.datetime.now()
                last = subtract_months(now, int(time))
            elif unit == 'year' or unit == 'years' or unit == 'yr':
                now = datetime.datetime.now()
                last = now - datetime.timedelta(days=365 * int(time))
            else:  # default last 1 month
                now = datetime.datetime.now()
                last = subtract_months(now, int(time))
            st_dt = datetime.datetime(last.year, last.month, last.day, last.hour, last.minute, last.second).astimezone()
            ed_dt = datetime.datetime(now.year, now.month, now.day, now.hour, now.minute, now.second).astimezone()
            start_date = st_dt.strftime(data_format)
            end_date = ed_dt.strftime(data_format)
            print(start_date)
            print(end_date)
            return start_date, end_date
        if data['state'][0] == 'current':
            if unit == 'hr' or unit == 'hour' or unit == 'hours' or unit == 'hrs':
                year = datetime.date.today().year
                month = datetime.date.today().month
                day = datetime.date.today().day
                hour = datetime.datetime.now().hour
                st_dt = datetime.datetime(year, month, day, hour, 0, 0).astimezone()
                ed_dt = datetime.datetime(year, month, day, hour, 59, 59).astimezone()
                start_date = st_dt.strftime(data_format)
                end_date = ed_dt.strftime(data_format)
                print(start_date)
                print(end_date)
                return start_date, end_date
            elif unit == 'day' or unit == 'days':
                year = datetime.date.today().year
                month = datetime.date.today().month
                day = datetime.date.today().day
                st_dt = datetime.datetime(year, month, day, 0, 0, 0).astimezone()
                ed_dt = datetime.datetime(year, month, day, 23, 59, 59).astimezone()
                start_date = st_dt.strftime(data_format)
                end_date = ed_dt.strftime(data_format)
                print(start_date)
                print(end_date)
                return start_date, end_date
            elif unit == 'week' or unit == 'weeks':
                today = datetime.date.today()
                current_week_start = today - datetime.timedelta(days=today.weekday())
                current_week_end = current_week_start + datetime.timedelta(days=6)
                st_dt = datetime.datetime(current_week_start.year, current_week_start.month, current_week_start.day, 0,
                                          0, 0).astimezone()
                ed_dt = datetime.datetime(current_week_end.year, current_week_end.month, current_week_end.day, 23, 59,
                                          59).astimezone()
                start_date = st_dt.strftime(data_format)
                end_date = ed_dt.strftime(data_format)
                print(start_date)
                print(end_date)
                return start_date, end_date
            elif unit == 'month' or unit == 'months':
                now = datetime.date.today()
                last_day = calendar.monthrange(now.year, now.month)[1]
                st_dt = datetime.datetime(now.year, now.month, 1, 0, 0, 0).astimezone()
                ed_dt = datetime.datetime(now.year, now.month, last_day, 23, 59, 59).astimezone()
                start_date = st_dt.strftime(data_format)
                end_date = ed_dt.strftime(data_format)
                print(start_date)
                print(end_date)
                return start_date, end_date
            elif unit == 'year' or unit == 'years' or unit == 'yr':
                now = datetime.date.today()
                st_dt = datetime.datetime(now.year, 1, 1, 0, 0, 0).astimezone()
                ed_dt = datetime.datetime(now.year, 12, 31, 23, 59, 59).astimezone()
                start_date = st_dt.strftime(data_format)
                end_date = ed_dt.strftime(data_format)
                print(start_date)
                print(end_date)
                return start_date, end_date
            else:
                now = datetime.date.today()
                last_day = calendar.monthrange(now.year, now.month)[1]
                st_dt = datetime.datetime(now.year, now.month, 1, 0, 0, 0).astimezone()
                ed_dt = datetime.datetime(now.year, now.month, last_day, 23, 59, 59).astimezone()
                start_date = st_dt.strftime(data_format)
                end_date = ed_dt.strftime(data_format)
                print(start_date)
                print(end_date)
                return start_date, end_date
        else:  # default current month report
            now = datetime.date.today()
            last_day = calendar.monthrange(now.year, now.month)[1]
            st_dt = datetime.datetime(now.year, now.month, 1, 0, 0, 0).astimezone()
            ed_dt = datetime.datetime(now.year, now.month, last_day, 23, 59, 59).astimezone()
            start_date = st_dt.strftime(data_format)
            end_date = ed_dt.strftime(data_format)
            print(start_date)
            print(end_date)
            return start_date, end_date


if __name__ == '__main__':
    data = {'year': ['2023', '2023'], 'time': ['2', '1', '3', '2']}
    data = {'month': ['january', 'mar'], 'year': ['2021', '2023']}
    data = {'month': ['february', 'march']}
    data = {'time': ['1', '28'], 'month': ['jan', 'sep'], 'year': ['2022', '2023']}
    data = {'state': ['today']}
    data = {'state': ['yesterday']}
    data = {'month': ['apr']}
    data = {'time': ['09'], 'month': ['dec'], 'year': ['2023']}
    data = {'state': ['last'], 'time': ['1'], 'unit': ['month']}
    data = {'state': ['last'], 'time': ['3'], 'unit': ['yr']}
    data = {'state': ['last'], 'unit': ['week']}
    data = {'state': ['last'], 'time': ['3']}
    data = {'state': ['last'], 'time': ['3'], 'unit': ['yr']}
    data = {'state': ['last'], 'unit': ['week']}
    data = {'state': ['last'], 'time': ['3']}
    data = {'state': ['current'], 'unit': ['yr']}
    data = {'state': ['current']}
    data = {'year': ['2023', '2023'], 'time': ['2', '1st', '3', '2nd']}
    data = {'time': ['9th'], 'month': ['dec'], 'year': ['2023']}
    data = {'time': ['1st', '28th'], 'month': ['jan', 'sep'], 'year': ['2022', '2023']}
    stat_date, end_date  = get_start_end_date(data)
    start_end_date_json = {'start_date':stat_date, 'end_date': end_date}
    print(start_end_date_json)
