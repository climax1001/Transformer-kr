f = open('../data/tmp/train.text', 'r')
while True:

        text = f.readline()
        print(text)
        if text == '':
            break