def init():
    global global_dict
    global_dict = {}


def set_value(key, value):
    global_dict[key] = value


def get_value(key, defValue=None):
    try:
        return global_dict[key]
    except KeyError:  # 查找字典的key不存在的时候触发
        return defValue