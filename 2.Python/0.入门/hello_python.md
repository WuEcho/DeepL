# Hello Python

## 标识符

只能由字母，下划线和数字组成，且数字不能作为开头，且python中标识符是区分大小写的，如`name`和`Name`是不一样的，且在命名变量的时候不能和关键字相同。

```
//错误示范
!age = 12
2age = 13
```

## 输出

- 简单输出：`print()`

```
print("python输出")
```

- 格式化输出：`print("%s" % value)`

```
password = input("请输入密码：")
print("您的密码是%s"% password)

t_id = input("学号：")
name = input("姓名：")
print("该学生的学号是：%s,姓名是：%s" %(t_id,name))

#输出
#学号： 12
#姓名： 李梅
#该学生的学号是：12,姓名是：李梅
#
```

## 单个赋值与多个赋值


```
//单个赋值
name = "李梅"

//多个赋值
id,name,age = 12,"李四",23
print("id:%d 姓名:%s 年龄：%d"%(id,name,age))
```


## 判断

- `if-else`

```
ticket = 1
if ticket == 1:
    print("有票")
else:
    print("无票")
    
/**输出内容为
有票
**/    
```

- `elif`

```
score = 70
if score >= 90 and score <= 100:
    print("A")
elif score >= 80 and score < 90:
    print("B")
elif score >= 70 and score < 80:
    print("C")
else:
    print("D")
    
/**输出内容为
C
**/    
```

## 循环
- `while`：不设置终止条件时会一直执行

```
i = 0
while i < 5:
    print("当前是执行第%d次"%(i+1))
    i+=1
/**输出内容
当前是执行第1次
当前是执行第2次
当前是执行第3次
当前是执行第4次
当前是执行第5次
**/    
```

- `for`

```
sum = 0
for i in [1,2,3,4]:
    sum += i
print(sum)    
```

- `break` 跳出循环


```
for i in [1,2,3,4,5]:
    print("------")
    if i == 3:
        break
    print(i)  
    
/**输出内容
------
1
------
2
------
**/    
```

- `continue` 跳出本次循环进入下次循环

```
for i in range(4):
    print("-----")
    if i == 2:
        continue
    print(i) 

/**输出内容
-----
0
-----
1
-----
-----
3
**/    
```



### 浮点数

精确控制浮点数 `decimal`

```python

import decimal
a = decimal.Decimal('0.1')
b = decimal.Decimal('0.2')
print(a+b)

# 0.3

```

### 数学公式

| 操作           | 结果                          |
|----------------|-------------------------------|
| x+y            | x加y的结果                    |
| x-y            | x减y的结果                    |
| x*y            | x乘以y的结果                  |
| x/y            | x除以y的结果                  |
| x//y           | x除以y的结果(地板除，除不尽的向下取整)|
| x%y            | x除以y的余数                  |
| abx(x)         | x的绝对值                     |
| int(x)         | 将x转成整数                   |
| float(x)       | 将x转换成浮点数               |
| complex(re,im) | 返回一个复数，re是实部，im是虚部 |
| c.conjugate()  | 返回c的共轭复数               |
| divmod(x,y)    | 返回(x//y,x%y)                |
| pow(x,y)       | 计算x的y次方                  |
| x ** y         | 计算x的y次方                  |

### bool 布尔值
被定义为False的对象：None和False
值为0的数字类型：0，0.0，0j，Decimal(0),Fraction(0,1)
空的序列和集合：‘’,(),[],{},set{},range(0)

### 运算符优先级

| 优先级 | 运算符                                                            | 描述                                      |
|--------|-------------------------------------------------------------------|-------------------------------------------|
| 1      | lambda                                                            | Lambda表达式                              |
| 2      | if-else                                                           | 条件表达式                                |
| 3      | or                                                                | 或                                        |
| 4      | and                                                               | 与                                        |
| 5      | not x                                                             | 非                                        |
| 6      | in，not in，is,is not,<,<=,>,>=,!=,==                               | 成员测试，同一性测试，比较                   |
| 7      |   |                                                               | 按位或                                    |
| 8      | ^                                                                 | 按位异或                                  |
| 9      | &                                                                 | 按位与                                    |
| 10     | <<,>>                                                             | 移位                                      |
| 11     | +，-                                                               | 加减法                                    |
| 12     | *，@，/,//,%                                                        |                                           |
| 13     | +x,-x,~x                                                          | 正号，负号，按位翻转                        |
| 14     | **                                                                | 指数                                      |
| 15     | await x                                                           | Await表达式                               |
| 16     | x[index],x[index:index],x(arguments...),x.attribute               | 下标，切片，函数调用，属性引用               |
| 17     | （expressions...）,[expressions...],{key:value...},{expressions...} | 绑定或元组显示，列表显示，字典显示，集合显示 |


