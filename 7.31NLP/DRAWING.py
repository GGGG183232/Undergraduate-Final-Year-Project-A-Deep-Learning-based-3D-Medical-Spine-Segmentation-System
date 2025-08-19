def decode_encoded_data(encoded_lines):
    # 存储每个节点的信息，索引对应行号（从 0 开始）
    nodes = []
    for line in encoded_lines:
        num, next_row = map(int, line.split())
        nodes.append((num, next_row))

    # 找到起始节点，即最后一个数指向的节点
    current_index = 0
    for i, (_, next_row) in enumerate(nodes):
        if next_row == 0:
            current_index = i
            break

    decoded = []
    while True:
        num, next_row = nodes[current_index]
        decoded.append(num)
        if next_row == 0:
            break
        # 下一个节点的行号需要减 1 来匹配索引
        current_index = next_row - 1

    # 逆序得到原始数据
    decoded.reverse()
    return ''.join(map(str, decoded))


# 存储控制台输入的编码数据
encoded = []
print("请逐行输入编码数据，输入空行结束输入：")
while True:
    line = input()
    if line == "":
        break
    encoded.append(line)

# 调用解码函数
decoded_result = decode_encoded_data(encoded)
print(decoded_result)
