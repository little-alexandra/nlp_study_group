#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# In[1]:


# 词典库
vocab = set([line.rstrip() for line in open('/Users/zhangning/py3/bin/python/纠错模型/vocab.txt')])


# In[16]:


# 需要生成所有候选集合
def generate_candidates(word):
    """
    word: 给定的输入（错误的输入） 
    返回所有(valid)候选集合
    """
    # 生成编辑距离为1的单词
    # 1.insert 2. delete 3. replace
    # appl: replace: bppl, cppl, aapl, abpl... 
    #       insert: bappl, cappl, abppl, acppl....
    #       delete: ppl, apl, app
    
    # 假设使用26个字符
    letters = 'abcdefghijklmnopqrstuvwxyz' 
    
    splits = [(word[:i], word[i:]) for i in range(len(word)+1)]
    # insert操作
    inserts = [L+c+R for L, R in splits for c in letters]
    # delete
    deletes = [L+R[1:] for L,R in splits if R]
    # replace
    replaces = [L+c+R[1:] for L,R in splits if R for c in letters]
    
    candidates = set(inserts+deletes+replaces)
    
    # 过滤调不存在于词典库里面的单词
    return [word for word in candidates if word in vocab] 
    
generate_candidates("appl")


# In[4]:


from nltk.corpus import reuters

# 读取语料库
categories = reuters.categories()
corpus = reuters.sents(categories=categories)


# In[7]:


# 构建语言模型: bigram
term_count = {}
bigram_count = {}
for doc in corpus:
    doc = ['<s>'] + doc
    for i in range(0, len(doc)-1):
        # bigram: [i,i+1]
        term = doc[i]
        bigram = doc[i:i+2]
        
        if term in term_count:
            term_count[term]+=1
        else:
            term_count[term]=1
        bigram = ' '.join(bigram)
        if bigram in bigram_count:
            bigram_count[bigram]+=1
        else:
            bigram_count[bigram]=1


# In[18]:


# 用户输错的概率统计 - channel probability
channel_prob = {}

for line in open('/Users/zhangning/py3/bin/python/纠错模型/spell-errors.txt'):
    items = line.split(":")
    correct = items[0].strip()
    mistakes = [item.strip() for item in items[1].strip().split(",")]
    channel_prob[correct] = {}
    for mis in mistakes:
        channel_prob[correct][mis] = 1.0/len(mistakes)


print("用户输错的概率",channel_prob)
print('----------------------------------------------------------')

# In[23]:


import numpy as np
V = len(term_count.keys())

file = open("/Users/zhangning/py3/bin/python/纠错模型/testdata.txt", 'r')
for line in file:
    items = line.rstrip().split('\t')
    line = items[2].split()
    # line = ["I", "like", "playing"]
    for word in line:
        if word not in vocab:
            # 需要替换word成正确的单词
            # Step1: 生成所有的(valid)候选集合
            candidates = generate_candidates(word)
            
            # 一种方式： if candidate = [], 多生成几个candidates, 比如生成编辑距离不大于2的
            # TODO ： 根据条件生成更多的候选集合
            if len(candidates) < 1:
                continue   # 不建议这么做（这是不对的） 
            probs = []
            # 对于每一个candidate, 计算它的score
            # score = p(correct)*p(mistake|correct)
            #       = log p(correct) + log p(mistake|correct)
            # 返回score最大的candidate
            for candi in candidates:
                prob = 0
                # a. 计算channel probability
                if candi in channel_prob and word in channel_prob[candi]:
                    prob += np.log(channel_prob[candi][word])
                else:
                    prob += np.log(0.0001)
                
                # b. 计算语言模型的概率
                idx = items[2].index(word)+1
                if items[2][idx - 1] in bigram_count and candi in bigram_count[items[2][idx - 1]]:
                    prob += np.log((bigram_count[items[2][idx - 1]][candi] + 1.0) / (
                            term_count[bigram_count[items[2][idx - 1]]] + V))
                # TODO: 也要考虑当前 [word, post_word]
                #   prob += np.log(bigram概率)
                
                else:
                    prob += np.log(1.0 / V)

                probs.append(prob)
                
            max_idx = probs.index(max(probs))
            print ("用户输入的单词，候选概率最大的单词",word, candidates[max_idx])


# In[ ]:


inverted_index = {}


