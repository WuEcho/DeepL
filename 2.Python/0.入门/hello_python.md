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

