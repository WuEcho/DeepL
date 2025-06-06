#分词方法：最大正向切分的第一种实现方式

import re
import time

#加载词典
def load_word_dict(path):
    max_word_length = 0
    word_dict = {}  #用set也是可以的。用list会很慢
    #打开文件 "r"是读取模式，"utf8"是编码格式
    #with 方式打开文件，自动关闭文件
    with open(path, encoding="utf8") as f:
        for line in f:
            #将每行文本按空白字符（空格、制表符等）分割成列表，并取第一个元素作为单词
            word = line.split()[0]
            word_dict[word] = 0
            max_word_length = max(max_word_length, len(word))
    return word_dict, max_word_length

#先确定最大词长度
#从长向短查找是否有匹配的词
#找到后移动窗口
def cut_method1(string, word_dict, max_len):
    words = []
    while string != '':
        lens = min(max_len, len(string))
        word = string[:lens]
        while word not in word_dict:
            if len(word) == 1:
                break
            word = word[:len(word) - 1] 
        words.append(word)
        string = string[len(word):]
    return words

#cut_method是切割函数
#output_path是输出路径
def main(cut_method, input_path, output_path):
    word_dict, max_word_length = load_word_dict("dict.txt")
    #打开文件 "w"是写入模式，"utf8"是编码格式
    writer = open(output_path, "w", encoding="utf8")
    start_time = time.time()
    with open(input_path, encoding="utf8") as f:
        for line in f:
            words = cut_method(line.strip(), word_dict, max_word_length)
            writer.write(" / ".join(words) + "\n")
    writer.close()
    print("耗时：", time.time() - start_time)
    return


string = "测试字符串"
word_dict, max_len = load_word_dict("dict.txt")
# print(cut_method1(string, word_dict, max_len))

main(cut_method1, "corpus.txt", "cut_method1_output.txt")
