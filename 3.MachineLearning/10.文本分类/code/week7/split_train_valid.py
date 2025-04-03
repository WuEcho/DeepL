import random
'''切割训练集和验证集'''
def split_file(file_path):
    # 打开文件，读取所有行，跳过第一行
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()[1:]
    # 随机打乱行顺序
    random.shuffle(lines)
    # 获取行数
    num_lines = len(lines)
    # 计算训练集行数
    num_train = int(0.8 * num_lines)

    # 将前80%的行作为训练集
    train_lines = lines[:num_train]
    # 将后20%的行作为验证集
    valid_lines = lines[num_train:]

    # 将训练集写入train_data.txt文件
    with open('train_data.txt', 'w', encoding='utf8') as f_train:
        f_train.writelines(train_lines)

    # 将验证集写入valid_data.txt文件
    with open('valid_data.txt', 'w', encoding='utf8') as f_valid:
        f_valid.writelines(valid_lines)

split_file(r'../文本分类练习数据集\文本分类练习.csv')