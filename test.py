

sk = open('train.skels', 'r')
l = []
while(True):
    line = sk.readlines()
    l.append(line)
    if line == []:
        break


    # print(line)
print(len(l[0][0].split(' ')))
# print("0 :" ,l[0])
# print("1 :" ,l[1])

sk.close()