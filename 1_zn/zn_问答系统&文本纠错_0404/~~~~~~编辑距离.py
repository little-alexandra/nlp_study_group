# !/usr/bin/env python
# coding: utf-8

# ### 编辑距离的计算
# 编辑距离可以用来计算两个字符串的相似度，它的应用场景很多，其中之一是拼写纠正（spell correction）。 编辑距离的定义是给定两个字符串str1和str2, 我们要计算通过最少多少代价cost可以把str1转换成str2.
#
# 举个例子：
#
# 输入:   str1 = "geek", str2 = "gesek"
# 输出:  1
# 插入 's'即可以把str1转换成str2
#
# 输入:   str1 = "cat", str2 = "cut"
# 输出:  1
# 用u去替换a即可以得到str2
#
# 输入:   str1 = "sunday", str2 = "saturday"
# 输出:  3
#
# 我们假定有三个不同的操作： 1. 插入新的字符   2. 替换字符   3. 删除一个字符。 每一个操作的代价为1.

# In[1]:


# 基于动态规划的解法
def edit_dist(str1, str2):
    # m，n分别字符串str1和str2的长度
    m, n = len(str1), len(str2)

    # 构建二位数组来存储子问题（sub-problem)的答案
    dp = [[0 for x in range(n + 1)] for x in range(m + 1)]

    # 利用动态规划算法，填充数组
    for i in range(m + 1):
        for j in range(n + 1):

            # 假设第一个字符串为空，则转换的代价为j (j次的插入)
            if i == 0:
                dp[i][j] = j

                # 同样的，假设第二个字符串为空，则转换的代价为i (i次的插入)
            elif j == 0:
                dp[i][j] = i

            # 如果最后一个字符相等，就不会产生代价
            elif str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]

                # 如果最后一个字符不一样，则考虑多种可能性，并且选择其中最小的值
            else:
                dp[i][j] = 1 + min(dp[i][j - 1],  # Insert
                                   dp[i - 1][j],  # Remove
                                   dp[i - 1][j - 1])  # Replace

    return dp[m][n]

print("编辑距离 ",edit_dist("appl","oppl"))

# ### 生成指定编辑距离的单词
# 给定一个单词，我们也可以生成编辑距离为K的单词列表。 比如给定 str="apple"，K=1, 可以生成“appl”, "appla", "pple"...等
# 下面看怎么生成这些单词。 还是用英文的例子来说明。 仍然假设有三种操作 - 插入，删除，替换

# In[5]:


def generate_edit_one(str):
    """
    给定一个字符串，生成编辑距离为1的字符串列表。
    """
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(str[:i], str[i:]) for i in range(len(str) + 1)]
    inserts = [L + c + R for L, R in splits for c in letters]
    deletes = [L + R[1:] for L, R in splits if R]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]

    # return set(splits)
    return set(inserts + deletes + replaces)


print((generate_edit_one("ab")))


# In[3]:


def generate_edit_two(str):
    """
    给定一个字符串，生成编辑距离不大于2的字符串
    """
    return [e2 for e1 in generate_edit_one(str) for e2 in generate_edit_one(e1)]


print((generate_edit_two("app")))

# ### 基于结巴（jieba）的分词。 Jieba是最常用的中文分词工具~

# In[1]:


# encoding=utf-8
import jieba

# 基于jieba的分词
seg_list = jieba.cut("本是同根生，相煎何太急", cut_all=False)
print("Default Mode: " + "/ ".join(seg_list))


jieba.add_word("相煎何")
seg_list = jieba.cut("本是同根生，相煎何太急", cut_all=False)
print("Default Mode: " + "/ ".join(seg_list))

# ### 判断一句话是否能够切分（被字典）

# In[5]:


dic = set(["贪心科技", "人工智能", "教育", "在线", "专注于"])


def word_break(str):
    could_break = [False] * (len(str) + 1)

    could_break[0] = True

    for i in range(1, len(could_break)):
        for j in range(0, i):
            if str[j:i] in dic and could_break[j] == True:
                could_break[i] = True

    return could_break[len(str)] == True


# In[6]:


assert word_break("贪心科技在线教育") == True
assert word_break("在线教育是") == False
assert word_break("") == True
assert word_break("在线教育人工智能") == True


# ### 思考题：给定一个词典和一个字符串，能不能返回所有有效的分割？ （valid segmentation)
# 比如给定词典：dic = set(["贪心科技", "人工智能", "教育", "在线", "专注于"， “贪心”])
# 和一个字符串 = “贪心科技专注于人工智能”
#
# 输出为：
# “贪心” “科技” “专注于” “人工智能”
# "贪心科技" “专注于” “人工智能”

# In[7]:


def all_possible_segmentations(str):
    segs = []

    return segs


# ### 停用词过滤
# 出现频率特别高的和频率特别低的词对于文本分析帮助不大，一般在预处理阶段会过滤掉。
# 在英文里，经典的停用词为 “The”, "an"....

# In[8]:


# 方法1： 自己建立一个停用词词典
stop_words = ["the", "an", "is", "there"]
# 在使用时： 假设 word_list包含了文本里的单词
word_list = ["we", "are", "the", "students"]
filtered_words = [word for word in word_list if word not in stop_words]
print(filtered_words)

# 方法2：直接利用别人已经构建好的停用词库
from nltk.corpus import stopwords

cachedStopWords = stopwords.words("english")

from nltk.stem.porter import *

stemmer = PorterStemmer()

test_strs = ['caresses', 'flies', 'dies', 'mules', 'denied',
             'died', 'agreed', 'owned', 'humbled', 'sized',
             'meeting', 'stating', 'siezing', 'itemization',
             'sensational', 'traditional', 'reference', 'colonizer',
             'plotted']

singles = [stemmer.stem(word) for word in test_strs]
print(' '.join(singles))  # doctest: +NORMALIZE_WHITESPACE

# ### 词袋向量： 把文本转换成向量 。 只有向量才能作为模型的输入。

# In[10]:


# 方法1： 词袋模型（按照词语出现的个数）
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
corpus = [
    'He is going from Beijing to Shanghai.',
    'He denied my request, but he actually lied.',
    'Mike lost the phone, and phone was in the car.',
]
X = vectorizer.fit_transform(corpus)

# In[11]:


print(X.toarray())
print(vectorizer.get_feature_names())

# In[12]:


# 方法2：词袋模型（tf-idf方法）
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(smooth_idf=False)
X = vectorizer.fit_transform(corpus)

# In[13]:


print(X.toarray())
print(vectorizer.get_feature_names())

# In[ ]:


# In[ ]:





