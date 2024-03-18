import datetime

def get_timestamp():
    '''
    Returns a timestamp in the format: "YYYY-MM-DD-HH-MM-SS"
    '''
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d-%H-%M-%S")