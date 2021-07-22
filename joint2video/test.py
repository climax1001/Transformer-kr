f = open('../data/phoenix2014T.train.de', 'r')
while True:

        text = f.readline()
        print(text)
        if text == '':
            break