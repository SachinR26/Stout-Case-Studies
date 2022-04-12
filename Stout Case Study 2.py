#!/usr/bin/env python
# coding: utf-8

# In[65]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[37]:


df = pd.read_csv('casestudy.csv')


# In[38]:


df_year2015 = df[df['year']==2015]
df_year2016 = df[df['year']==2016]
df_year2017 = df[df['year']==2017]


# In[39]:


df_year2017


# In[40]:


Revenue2015 = df_year2015['net_revenue'].sum()
Revenue2016 = df_year2016['net_revenue'].sum()
Revenue2017 = df_year2017['net_revenue'].sum()


# In[41]:


print("Revenue in 2015:", Revenue2015)
print("Revenue in 2016:", Revenue2016)
print("Revenue in 2017:", Revenue2017)


# In[42]:


df_new2016 = pd.merge(df_year2015, df_year2016, on='customer_email', how='outer',indicator=True)
df_new2016 = df_new2016[df_new2016['year_y']==2016]
df_new2016 = df_new2016[df_new2016['year_x'].isna()]
NewCustomer_Revenue_2016 = df_new2016['net_revenue_y'].sum()


# In[43]:


print("Revenue of New Customers in 2016:",NewCustomer_Revenue_2016)


# In[44]:


df_new2017 = pd.merge(df_year2016, df_year2017, on='customer_email', how='outer',indicator=True)
df_new2017 = df_new2017[df_new2017['year_y']==2017]
df_new2017 = df_new2017[df_new2017['year_x'].isna()]
NewCustomer_Revenue_2017 = df_new2017['net_revenue_y'].sum()


# In[45]:


print("Revenue of New Customers in 2017:",NewCustomer_Revenue_2017)


# In[46]:


df_new2017 = pd.merge(df_year2016, df_year2017, on='customer_email', how='outer',indicator=True)


# In[47]:


df_new2017 = df_new2017[df_new2017['_merge']=='both']


# In[48]:


df_existingdiff2017 = pd.DataFrame(columns=['customer_email','net_growth'])
df_existingdiff2017['net_growth'] = df_new2017['net_revenue_y'] - df_new2017['net_revenue_x']
df_existingdiff2017['customer_email'] = df_new2017['customer_email']


# In[49]:


print("Exisiting customer growth in 2017", df_existingdiff2017)


# In[50]:


df_new2016 = pd.merge(df_year2015, df_year2016, on='customer_email', how='outer',indicator=True)
df_new2016 = df_new2016[df_new2016['_merge']=='both']


# In[51]:


df_existingdiff2016 = pd.DataFrame(columns=['customer_email','net_growth'])
df_existingdiff2016['net_growth'] = df_new2016['net_revenue_y'] - df_new2016['net_revenue_x']
df_existingdiff2016['customer_email'] = df_new2016['customer_email']


# In[52]:


print("Exisiting customer growth in 2016", df_existingdiff2016)


# In[53]:


Revenue_attrition2017 = df_existingdiff2017['net_growth'].sum()
Revenue_attrition2016 = df_existingdiff2016['net_growth'].sum()


# In[54]:


print("Revenue lost from attrition in 2017:", Revenue_attrition2017)
print("Revenue lost from attrition in 2016:", Revenue_attrition2016)


# In[55]:


Ex_custrev_2017 = df_new2017['net_revenue_y'].sum()
Ex_custrev_2016 = df_new2017['net_revenue_x'].sum()


# In[56]:


print("Revenue in 2017(Current Year):", Ex_custrev_2017)
print("Revenue in 2016(Previous Year):", Ex_custrev_2016)


# In[57]:


Ex_custrev_2016 = df_new2016['net_revenue_y'].sum()
Ex_custrev_2015 = df_new2016['net_revenue_x'].sum()


# In[58]:


print("Revenue in 2016(Current Year):", Ex_custrev_2016)
print("Revenue in 2015(Previous Year):", Ex_custrev_2015)


# In[59]:


print("Total Customers in 2017:", len(df_year2017))
print("Total Customers in 2016:", len(df_year2016))
print("Total Customers in 2015:", len(df_year2015))


# In[60]:


df_custlist2017 = pd.merge(df_year2016, df_year2017, on='customer_email', how='outer',indicator=True)
df_custlist2016 = pd.merge(df_year2015, df_year2016, on='customer_email', how='outer',indicator=True)


# In[61]:


df_newcust2017 = df_custlist2017[df_custlist2017['_merge']=='right_only']
df_lostcust2017 = df_custlist2017[df_custlist2017['_merge']=='left_only']


# In[62]:


print("New Customers in 2017", df_newcust2017['customer_email'])
print("Lost Customers in 2017",df_lostcust2017['customer_email'])


# In[63]:


df_newcust2016 = df_custlist2016[df_custlist2016['_merge']=='right_only']
df_lostcust2016 = df_custlist2016[df_custlist2016['_merge']=='left_only']


# In[64]:


print("New Customers in 2016", df_newcust2016['customer_email'])
print("Lost Customers in 2016",df_lostcust2016['customer_email'])


# In[88]:


Revenue = [Revenue2015,Revenue2016,Revenue2017]
RevenueYear = ['2015', '2016', '2017']


# In[13]:


df_new2017


# In[91]:


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(RevenueYear, Revenue)
ax.set_ylabel('Revenue in Million')
ax.set_xlabel('Year')
plt.show() 


# In[74]:


X = ['2016','2017']
X_axis = np.arange(len(X))
  
plt.bar(X_axis - 0.2, (len(df_newcust2016), len(df_newcust2017)), 0.4, label = 'New Customers', color=['red'])
plt.bar(X_axis + 0.2, (len(df_year2016), len(df_year2017)), 0.4, label = 'Total Customers', color=['black'])
  
plt.xticks(X_axis, X)
plt.xlabel("Years")
plt.ylabel("Number of Customers")
plt.title("Total Customers vs New customers by Year")
plt.legend()
plt.show()


# In[76]:


X = ['2016','2017']
X_axis = np.arange(len(X))
  
plt.bar(X_axis - 0.2, (Ex_custrev_2016, Ex_custrev_2017), 0.4, label = 'Existing Customer Revenue', color=['red'])
plt.bar(X_axis + 0.2, (Revenue2016, Revenue2017), 0.4, label = 'Total Revenue', color=['black'])
  
plt.xticks(X_axis, X)
plt.xlabel("Years")
plt.ylabel("Revenue Earned")
plt.title("Total Revenue vs Existing Customer Revenue by Year")
plt.legend()
plt.show()


# In[77]:


X = ['2016','2017']
X_axis = np.arange(len(X))
  
plt.bar(X_axis - 0.2, (NewCustomer_Revenue_2016, NewCustomer_Revenue_2017), 0.4, label = 'New Customer Revenue', color=['red'])
plt.bar(X_axis + 0.2, (Revenue2016, Revenue2017), 0.4, label = 'Total Revenue', color=['black'])
  
plt.xticks(X_axis, X)
plt.xlabel("Years")
plt.ylabel("Revenue Earned")
plt.title("Total Revenue vs New Customer Revenue by Year")
plt.legend()
plt.show()

