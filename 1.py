import os

filepath = 'layouts\mediumClassic.lay'

# 打印完整路径
print("完整路径：", os.path.abspath(filepath))

# 打印当前工作目录
print("当前工作目录：", os.getcwd())

# 尝试打开文件
try:
    f = open(filepath)
    print("文件打开成功")
except FileNotFoundError:
    print("无法找到文件，请检查路径和工作目录")
except Exception as e:
    print("打开文件时出错：", e)
finally:
    f.close()
