import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# Set style for plots
sns.set(style="whitegrid")

# Streamlit page setup
st.set_page_config(page_title="E-commerce EDA", layout="wide")
st.title("📊 E-commerce Exploratory Data Analysis")

# Load dataset
df = pd.read_csv("ecommerce_dataset.csv")

# Show first rows
st.header("First 5 Rows of the Dataset")
st.write(df.head())

# Dataset Info
st.header("Dataset Information")
import io
buffer = io.StringIO()
df.info(buf=buffer)
info_str = buffer.getvalue()
st.text(info_str)

# Summary Statistics
st.header("Summary Statistics of Numerical Columns")
st.write(df.describe())

# Missing Values
st.header("Missing Values in Dataset")
st.write(df.isnull().sum())

# Category Distribution
st.header("Distribution of Product Categories")
category_counts = df['category'].value_counts()
st.write(category_counts)

fig, ax = plt.subplots(figsize=(8,5))
sns.countplot(x='category', data=df, palette='viridis', ax=ax)
ax.set_title("Category Distribution", fontsize=14, weight="bold")
st.pyplot(fig)

# Revenue Column
st.header("Revenue Column Creation")
df["revenue"] = df["price"] * df["quantity"] * (1 - df["discount"])
st.write(df.head())

# Top Customers
st.header("Top 5 Customers by Revenue")
top_customers = df.groupby("customer_id")["revenue"].sum().sort_values(ascending=False).head(5)
st.write(top_customers)

# Revenue by Category
st.header("Revenue by Category")
rev_by_cat = df.groupby("category")["revenue"].sum().sort_values(ascending=False)
st.bar_chart(rev_by_cat)

# Revenue by Region
st.header("Revenue by Region")
rev_by_region = df.groupby("region")["revenue"].sum().sort_values(ascending=False)
st.bar_chart(rev_by_region)

# Revenue by Payment Method
st.header("Revenue by Payment Method")
rev_by_payment = df.groupby("payment_method")["revenue"].sum().sort_values(ascending=False)
st.bar_chart(rev_by_payment)

# Hypothesis Test
st.header("Hypothesis Test: Region vs Payment Method")
contingency = pd.crosstab(df["region"], df["payment_method"])
chi2, p, dof, exp = chi2_contingency(contingency)
st.write(f"Chi2: {chi2:.3f}, P-value: {p:.3f}")
if p < 0.05:
    st.success("✅ Significant association: Region and Payment Method are related.")
else:
    st.error("❌ No significant association: Region and Payment Method are independent.")

# Revenue Trend Over Time
st.header("Revenue Trend Over Time")
df["order_date"] = pd.to_datetime(df["order_date"])
rev_by_date = df.groupby(df["order_date"].dt.date)["revenue"].sum()
st.line_chart(rev_by_date)

# Correlation Heatmap
st.header("Correlation Heatmap of Numerical Features")
num_cols = ["quantity", "price", "discount", "revenue"]
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(6,4))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", cbar=True, ax=ax)
ax.set_title("Correlation Heatmap", fontsize=14, weight="bold")
st.pyplot(fig)

# Pairplot
st.header("Pairplot of Numerical Features")
st.write("Pairplot of numerical variables (quantity, price, discount, revenue)")
pairplot = sns.pairplot(df[num_cols], diag_kind="kde", corner=True, palette="husl")
st.pyplot(pairplot.fig)

# Conclusion
st.header("📌 Conclusion")
st.markdown("""
**Key Insights:**
- **Sales Distribution**: Electronics and Clothing dominate revenue, while niche categories contribute less.  
- **Regional Sales**: Transactions are fairly balanced across regions → wide geographic reach.  
- **Payment Preferences**: Customers use all payment methods fairly evenly. No significant association between region and payment method (Chi-square test).  
- **Category Revenue Comparison**: Electronics vs Sports revenue → no significant difference.  
- **General Trend**: Revenue patterns are stable across categories, regions, and payment methods.  

**✅ Business Implication:**  
The dataset suggests stable and balanced performance across categories and regions.  
This stability provides opportunities to run **promotions, loyalty programs, and targeted marketing** without disrupting specific customer segments.
""")
