#!/usr/bin/env python
# coding: utf-8

# ### 例子： Fibonanci number (斐波那契数)
# 序列一次为 1，1，2，3，5，8，13，21，....
# 问题： 怎么求出序列中第N个数？ 

# In[2]:


def fib(n):
    """
    我们假设 n=>0 
    """
    if n < 3: 
        return 1
    else:
        return fib(n-2)+fib(n-1)


# In[ ]:





# In[ ]:


print (fib(50))  


# In[3]:


# 利用动态规划来解决此问题
import numpy as np

def fib_dp(n):
    tmp = np.zeros(n)  # 数组
    tmp[0] = 1
    tmp[1] = 1
    for i in range(2,n):
        tmp[i] = tmp[i-2] + tmp[i-1]
    
    return tmp[n-1]

时间复杂度 o(n)   空间复杂度： o(n)


# In[5]:


print (int(fib_dp(100)))


# In[ ]:


def fib_dp2(n):
    a,b=1,1
    c = 0
    for i in range(2,n):
        c = a + b
        a = b
        b = c
    return c


# In[ ]:


print (fib_dp2(10))


# In[ ]:





