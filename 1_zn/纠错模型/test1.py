channel_prob = {}

for line in open('/Users/zhangning/py3/bin/python/纠错模型/spell-errors.txt'):
    items = line.split(":")
    correct = items[0].strip()
    mistakes = [item.strip() for item in items[1].strip().split(",")]
    channel_prob[correct] = {}
    for mis in mistakes:
        channel_prob[correct][mis] = 1.0/len(mistakes)

##print(channel_prob)
print(correct)