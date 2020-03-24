#!/usr/bin/env python
# coding: utf-8

# # Python相似度计算
# https://blog.csdn.net/Yellow_python/article/details/81069692

# In[1]:


from warnings import filterwarnings
filterwarnings('ignore')  # 不打印警告
import numpy as np
from pandas import DataFrame


# # 语料


poem1 = '''《将进酒》——李白
君不见黄河之水天上来，奔流到海不复回。君不见高堂明镜悲白发，朝如青丝暮成雪。
人生得意须尽欢，莫使金樽空对月。天生我材必有用，千金散尽还复来。
烹羊宰牛且为乐，会须一饮三百杯。岑夫子，丹丘生，将进酒，杯莫停。与君歌一曲，请君为我倾耳听。
钟鼓馔玉不足贵，但愿长醉不复醒。古来圣贤皆寂寞，惟有饮者留其名。
陈王昔时宴平乐，斗酒十千恣欢谑。主人何为言少钱，径须沽取对君酌。
五花马，千金裘，呼儿将出换美酒，与尔同销万古愁。'''.replace('\n', '')
poem2 = '''《惜樽空》——李白
君不见黄河之水天上来，奔流到海不复回。君不见床头明镜悲白发，朝如青云暮成雪。
人生得意须尽欢，莫使金樽空对月。天生吾徒有俊才，千金散尽还复来。
烹羊宰牛且为乐，会须一饮三百杯。岑夫子，丹丘生，与君哥一曲，请君为我倾。
钟鼓玉帛岂足贵，但用长醉不复醒。古来贤圣皆死尽，惟有饮者留其名。
陈王昔时宴平乐，斗酒十千恣欢谑。主人何为言少钱，径须沽取对君酌。
五花马，千金裘，呼儿将出换美酒，与尔同销万古愁。'''.replace('\n', '')
ls1, ls2 = list(poem1), list(poem2)
##print(ls1,ls2)

# # 交集 / 并集


s1, s2 = set(ls1), set(ls2)
len(s1 & s2) / len(s1 | s2)

print("交并集",s1,s2)
print("两段诗词长度合集/交集的比值",len(s1 & s2) / len(s1 | s2))



# # 编辑距离
# 针对二个字符串的差异程度的量化量测，量测方式是看至少需要多少次的处理才能将一个字符串变成另一个字符串。

# #### 莱文斯坦距离
# 允许`删除、加入、取代字符串`

# In[4]:


def edit_distance_matrix(s1, s2):
    l1, l2 = len(s1) + 1, len(s2) + 1
    matrix = np.zeros((l1, l2), dtype=int)
    for i in range(l1):
        matrix[i, 0] = i
    for j in range(l2):
        matrix[0, j] = j
    for i in range(1, l1):
        for j in range(1, l2):
            delta = 0 if s1[i - 1] == s2[j - 1] else 1
            matrix[i, j] = min(matrix[i - 1, j - 1] + delta,
                               matrix[i - 1, j] + 1,
                               matrix[i, j - 1] + 1)
    return matrix
s1, s2 = 'abc', 'bacd'
DataFrame(edit_distance_matrix(s1, s2), list(' ' + s1), list(' ' + s2))


# #### 基于编辑距离的相似度百分比

# In[5]:


edit_distance = edit_distance_matrix(poem1, poem2)[-1, -1]

print("~~~~~~~~~~~~~量测方式是看至少需要多少次的处理才能将一个字符串变成另一个字符串~~~~~~~~~~~~~~~~")

print("莱文斯坦编辑距离",edit_distance)
print("莱文斯坦编辑距离百分比",1 - edit_distance / (len(poem1 + poem2) / 2))


# #### Damerau-Levenshtein
# 允许`删除、加入、取代字符串、字符转置`

# In[6]:


from nltk import edit_distance
print("nltk_Damerau-Levenshtein编辑距离",edit_distance(s1, s2))
print("nltk_Damerau-Levenshtein",edit_distance(s1, s2, transpositions=True))  # 允许字符转置


# # 欧氏距离 余弦距离

# In[7]:


word2id = {w: i for i, w in enumerate(set(ls1 + ls2))}
length = len(word2id)
vec1, vec2 = np.zeros(length), np.zeros(length)
for w in ls1:
    vec1[word2id[w]] += 1
for w in ls2:
    vec2[word2id[w]] += 1


# #### 欧氏距离

# In[8]:
print("-----------------------------------------------")
print("文本相似度")

print("欧式距离",np.linalg.norm(vec1 - vec2)) # np.sqrt(np.sum(np.square(vec1 - vec2)))


# #### 余弦距离

print("余弦距离",1 - vec1 @ vec2 / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


# #### 调包实现

# In[10]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
vec1, vec2 = CountVectorizer(token_pattern='.').fit_transform([poem1, poem2])
print("调包实现欧式距离",euclidean_distances(vec1, vec2))
print("调包实现余弦距离",cosine_distances(vec1, vec2))
print("-----------------------------------------------")

# # 最长公共子串
# Longest Common Substring

# In[11]:


def lcs_matrix(s1, s2):
    l1, l2 = len(s1), len(s2)
    matrix = [[0 for j in range(l2 + 1)] for i in range(l1 + 1)]
    for i in range(l1):
        for j in range(l2):
            if s1[i] == s2[j]:
                matrix[i + 1][j + 1] = matrix[i][j] + 1
    return DataFrame(matrix, list(' ' + s1), list(' ' + s2))
print(lcs_matrix('cdef', 'abcde'),"举例_最长公共子串--cdef & abcde")


# In[12]:

print("-----------------------------------------------")
def lcs(s1, s2):
    l1, l2 = len(s1), len(s2)
    matrix = np.zeros((l1 + 1, l2 + 1), dtype=int)
    max_len = 0  # 最长匹配的长度
    p = 0  # 最长匹配对应在s1中的最后一位
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                matrix[i + 1, j+1] = matrix[i, j] + 1
                max_len, p = max([(max_len, p), (matrix[i + 1, j + 1], i + 1)])
    return s1[p - max_len: p]  # 返回最长子串及其长度
print("两端诗词的最长公共子串: ",lcs(poem1, poem2))

print("-----------------------------------------------")
# # 最长公共子序列
# The longest common subsequence

# In[13]:


s1, s2 = 'ab_abc', 'abc_bcd'
l1, l2 = len(s1), len(s2)
# 生成字符串长度+1的零矩阵，保存对应位置匹配的结果
m = [[0 for j in range(l2 + 1)] for i in range(l1 + 1)]
# 记录转移方向
d = [['' for j in range(l2 + 1)] for i in range(l1 + 1)]
for p1 in range(l1):
    for p2 in range(l2):
        # 字符匹配成功，则该位置的值为左上方的值加1
        if s1[p1] == s2[p2]:
            m[p1 + 1][p2 + 1] = m[p1][p2] + 1
            d[p1 + 1][p2 + 1] = 'ok'
        # 左值大于上值，则该位置的值为左值，并标记回溯时的方向
        elif m[p1 + 1][p2] > m[p1][p2 + 1]:
            m[p1 + 1][p2 + 1] = m[p1 + 1][p2]
            d[p1 + 1][p2 + 1] = '←'
        # 上值大于左值，则该位置的值为上值，并标记方向↑
        else:
            m[p1 + 1][p2 + 1] = m[p1][p2 + 1]
            d[p1 + 1][p2 + 1] = '↑'


# In[14]:


DataFrame(m, list(' ' + s1), list(' ' + s2))


# In[15]:


DataFrame(d, list(' ' + s1), list(' ' + s2))


# In[16]:


def lcs(s1, s2):
    l1, l2 = len(s1), len(s2)
    # 生成字符串长度+1的零矩阵，保存对应位置匹配的结果
    m = np.zeros((l1 + 1, l2 + 1))
    # 记录转移方向
    d = np.empty_like(m, dtype=str)
    for i in range(l1):
        for j in range(l2):
            # 字符匹配成功，则该位置的值为左上方的值加1
            if s1[i] == s2[j]:
                m[i + 1, j + 1] = m[i, j] + 1
                d[i + 1, j + 1] = 'O'
            # 左值大于上值，则该位置的值为左值，并标记回溯时的方向
            elif m[i + 1, j] > m[i, j + 1]:
                m[i + 1, j + 1] = m[i + 1, j]
                d[i + 1, j + 1] = '←'
            # 上值大于左值，则该位置的值为上值，并标记方向↑
            else:
                m[i + 1, j + 1] = m[i, j + 1]
                d[i + 1, j + 1] = '↑'
    s = []
    while m[l1, l2]:  # 不为空时
        c = d[l1, l2]
        if c == 'O':  # 匹配成功，插入该字符，并向左上角找下一个
            s.append(s1[l1 - 1])
            l1 -= 1
            l2 -= 1
        if c == '←':  # 根据标记，向左找下一个
            l2 -= 1
        if c == '↑':  # 根据标记，向上找下一个
            l1 -= 1
    s.reverse()
    return ''.join(s)
print("标记对应位置————————",lcs(poem1, poem2))


# # TfIdf文本相似度

# #### 语料加载

# In[17]:
print("-----------------------------------------------")
print("----所有古诗的语料加载 ------------------")
with open('/Users/zhangning/py3/bin/python/PyProjects/NLP/文本相似度/古诗.txt', encoding='utf-8') as f:
    seqs = f.read().split()
q_ls = [i.split('|')[1] for i in seqs]
a_ls = [i.split('|')[0] for i in seqs]


# #### 训练tfidf向量转换器

# In[18]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
vectorizer = TfidfVectorizer(token_pattern='.')
vectorizer.fit(q_ls)
vec1, vec2 = vectorizer.transform([poem1, poem2])
print("李白的两首诗词的向量转换",vec1,vec2)
print("李白的两首诗词的余弦相似度",cosine_similarity(vec1, vec2))


# In[ ]:
print("-----------------------------------------------------")
print("语料的向量转换")
print("---------------------没看懂，谁跟谁的余弦相似度，语料里每个词跟后两个词的么？到排表在哪儿")
X = vectorizer.transform(q_ls)
def ask(q, n=2):
    q = vectorizer.transform([q])  # tfidf向量化
    indexs = cosine_similarity(X, q).reshape(-1)  # 余弦相似度
    indexs = np.argsort(-indexs)  # 按索引倒排
    return [a_ls[i] for i in indexs[:n]]
for _ in range(5):
    q = input('输入：').strip()
    for e, i in enumerate(ask(q)):
        print(e, i)


# # 词向量文本相似度

# In[ ]:

print("-----------------------------------------------------")
print("-----------分布是向量的文本相似度的计算----------------------")
from gensim.models import Word2Vec
ls_of_words = [list(i) for i in q_ls]
model = Word2Vec(ls_of_words)
w2i = {w: i for i, w in enumerate(model.wv.index2word, 1)}
vectors = np.concatenate((np.zeros((1, 100)), model.wv.vectors), axis=0)
w2v = lambda w: vectors[w2i.get(w, 0)]


# In[ ]:


vec1 = np.mean([w2v(w) for w in poem1], axis=0)
vec2 = np.mean([w2v(w) for w in poem2], axis=0)


# In[ ]:


print(cosine_similarity([vec1], [vec2]))
vec1 @ vec2 / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





