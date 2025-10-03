#!/usr/bin/env python
# coding: utf-8

# In[15]:


from IPython.display import display, Markdown
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for plots
sns.set(style="whitegrid")

# Load dataset
df = pd.read_csv("ecommerce_dataset.csv")

# Function to make big bold headings
def heading(text):
    display(Markdown(f"## **{text}**"))


# In[17]:


heading("First 5 Rows of the Dataset")
display(df.head())


# In[19]:


heading("Dataset Information")
df.info()


# In[21]:


heading("Summary Statistics of Numerical Columns")
display(df.describe())


# In[23]:


heading("Missing Values in Dataset")
display(df.isnull().sum())


# In[25]:


heading("Distribution of Product Categories")

# Counts
category_counts = df['category'].value_counts()
display(category_counts)

# Plot
plt.figure(figsize=(8,5))
sns.countplot(x='category', data=df, palette='viridis')
plt.title("Category Distribution", fontsize=14, weight="bold")
plt.show()


# In[27]:


heading("Revenue Column Creation")

# Revenue = price * quantity * (1 - discount)
df["revenue"] = df["price"] * df["quantity"] * (1 - df["discount"])

# Show first few rows with revenue
display(df.head())


# In[29]:


heading("Top 5 Customers by Revenue")

top_customers = df.groupby("customer_id")["revenue"].sum().sort_values(ascending=False).head(5)
display(top_customers)


# In[31]:


heading("Revenue by Category")

rev_by_cat = df.groupby("category")["revenue"].sum().sort_values(ascending=False)
display(rev_by_cat)

plt.figure(figsize=(8,5))
sns.barplot(x=rev_by_cat.index, y=rev_by_cat.values, palette="plasma")
plt.title("Revenue by Product Category", fontsize=14, weight="bold")
plt.ylabel("Revenue")
plt.xlabel("Category")
plt.show()


# In[33]:


heading("Revenue by Region")

rev_by_region = df.groupby("region")["revenue"].sum().sort_values(ascending=False)
display(rev_by_region)

plt.figure(figsize=(6,5))
sns.barplot(x=rev_by_region.index, y=rev_by_region.values, palette="coolwarm")
plt.title("Revenue by Region", fontsize=14, weight="bold")
plt.ylabel("Revenue")
plt.xlabel("Region")
plt.show()


# In[35]:


heading("Revenue by Payment Method")

rev_by_payment = df.groupby("payment_method")["revenue"].sum().sort_values(ascending=False)
display(rev_by_payment)

plt.figure(figsize=(7,5))
sns.barplot(x=rev_by_payment.index, y=rev_by_payment.values, palette="Set2")
plt.title("Revenue by Payment Method", fontsize=14, weight="bold")
plt.ylabel("Revenue")
plt.xlabel("Payment Method")
plt.show()


# In[39]:


heading("Hypothesis Test: Region vs Payment Method")

from scipy.stats import chi2_contingency

contingency = pd.crosstab(df["region"], df["payment_method"])
chi2, p, dof, exp = chi2_contingency(contingency)

display(Markdown(f"Chi2: {chi2:.3f}, P-value: {p:.3f}"))

if p < 0.05:
    display(Markdown("✅ Significant association: Region and Payment Method are related."))
else:
    display(Markdown("❌ No significant association: Region and Payment Method are independent."))


# In[41]:


heading("Revenue Trend Over Time")

df["order_date"] = pd.to_datetime(df["order_date"])
rev_by_date = df.groupby(df["order_date"].dt.date)["revenue"].sum()

plt.figure(figsize=(10,5))
rev_by_date.plot(kind="line", marker="o", color="green")
plt.title("Revenue Trend Over Time", fontsize=14, weight="bold")
plt.xlabel("Date")
plt.ylabel("Revenue")
plt.show()


# In[43]:


heading("Correlation Heatmap of Numerical Features")

# Select numerical columns
num_cols = ["quantity", "price", "discount", "revenue"]

# Correlation matrix
corr = df[num_cols].corr()

plt.figure(figsize=(6,4))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
plt.title("Correlation Heatmap", fontsize=14, weight="bold")
plt.show()


# In[49]:


heading("Pairplot of Numerical Features")

sns.pairplot(df[num_cols], diag_kind="kde", corner=True, palette="husl")
plt.suptitle("Pairplot of Numerical Variables", y=1.02, fontsize=14, weight="bold")
plt.show()


# # 📌 Conclusion
# 
# The analysis of the **E-commerce Dataset** highlights the following key insights:
# 
# - **Sales Distribution**: A few categories such as *Electronics* and *Clothing* dominate the revenue, while niche categories contribute relatively less.  
# - **Regional Sales**: Transactions are fairly balanced across regions, showing that the platform has a wide geographic reach without major regional biases.  
# - **Payment Preferences**: Customers use all payment methods fairly evenly. Our statistical test showed **no significant association** between region and payment method.  
# - **Category Revenue Comparison**: A hypothesis test between *Electronics* and *Sports* revenue revealed **no significant difference**, meaning these categories generate similar revenue patterns.  
# - **General Trend**: Revenue patterns remain consistent across product types, payment methods, and regions.  
# 
# ---
# 
# ### ✅ Business Implication
# The dataset suggests that the e-commerce platform has **stable and balanced performance** across categories and regions.  
# This stability provides an excellent opportunity to **experiment with promotions, loyalty programs, or targeted marketing** without fear of disrupting a specific region or payment channel.  
# 
# 

# In[ ]:




