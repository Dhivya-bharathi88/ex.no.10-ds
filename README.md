# Ex:10 Data Science Process on Complex Dataset
# AIM:
To Perform Data Science Process on a complex dataset and save the data to a file

# ALGORITHM:
STEP 1 Read the given Data

STEP 2 Clean the Data Set using Data Cleaning Process

STEP 3 Apply Feature Generation/Feature Selection Techniques on the data set

STEP 4 Apply EDA /Data visualization techniques to all the features of the data set

# CODE:

# Data Cleaning Process:

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib as plt

from google.colab import files

uploaded = files.upload()

df = pd.read_csv("Bank Customer Churn Prediction.csv")

df.head(10)

df.info()

df.describe()

df.isnull().sum()

# Handling Outliers:
q1 = df['balance'].quantile(0.25)

q3 = df['balance'].quantile(0.75)

IQR = q3 - q1

print("First quantile:", q1, " Third quantile:", q3, " IQR:", IQR, "\n")

lower = q1 - 1.5 * IQR

upper = q3 + 1.5 * IQR

outliers = df[(df['balance'] >= lower) & (df['balance'] <= upper)]

from scipy.stats import zscore

z = outliers[(zscore(outliers['balance']) < 3)]

print("Cleaned Data: \n")

print(z)

# EDA Techniques:

df.skew()

df.kurtosis()

~df.duplicated()

df1=df[~df.duplicated()]

df1
sns.boxplot(x="balance",data=df)

sns.boxplot(x="age",data=df)

sns.countplot(x="age",data=df)

sns.distplot(df["age"])

sns.histplot(df["age"])

sns.displot(df["balance"])

sns.scatterplot(x=df['credit_score'],y=df['balance'])

import matplotlib.pyplot as plt

states=df.loc[:,["country","credit_score"]]

states=states.groupby(by=["country"]).sum().sort_values(by="credit_score")

plt.figure(figsize=(17,7))

sns.barplot(x=states.index,y="credit_score",data=states)

plt.xlabel=("country")

plt.ylabel=("credit_score")

plt.show()

import matplotlib.pyplot as plt

states=df.loc[:,["gender","credit_score"]]

states=states.groupby(by=["gender"]).sum().sort_values(by="credit_score")

plt.figure(figsize=(17,7))

sns.barplot(x=states.index,y="credit_score",data=states)

plt.xlabel=("gender")

plt.ylabel=("credit_score")

plt.show()

df.corr()

sns.heatmap(df.corr(),annot=True)

# Feature Generation:

from sklearn.preprocessing import LabelEncoder,OrdinalEncoder

le=LabelEncoder()

df['credit']=le.fit_transform(df['credit_score'])

df

le=LabelEncoder()

df['customer']=le.fit_transform(df['customer_id'])

df

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from google.colab import files

uploaded = files.upload()

from sklearn.preprocessing import OneHotEncoder

df1 = pd.read_csv("Bank Customer Churn Prediction.csv")

ohe=OneHotEncoder(sparse=False)

enc=pd.DataFrame(ohe.fit_transform(df1[['country']]))

df1=pd.concat([df1,enc],axis=1)

df1

import statsmodels.api as sm

import scipy.stats as stats

from sklearn.preprocessing import QuantileTransformer

sm.qqplot(df1['credit_score'],fit=True,line='45')

plt.show()

sm.qqplot(df1['estimated_salary'],fit=True,line='45')

plt.show()

import numpy as np

from sklearn.preprocessing import PowerTransformer

transformer=PowerTransformer("yeo-johnson")

df1['active_member']=pd.DataFrame(transformer.fit_transform(df1[['active_member']]))
sm.qqplot(df1['active_member'],line='45')

plt.show()

transformer=PowerTransformer("yeo-johnson")

df1['credit_card']=pd.DataFrame(transformer.fit_transform(df1[['credit_card']]))

sm.qqplot(df1['credit_card'],line='45')

plt.show()

qt=QuantileTransformer(output_distribution='normal')

df1['credit_score']=pd.DataFrame(qt.fit_transform(df1[['credit_score']]))

sm.qqplot(df1['credit_score'],line='45')

plt.show()

df1.drop([0,1,2],axis=1, inplace=True)

df1

# Data Visualization:

sns.barplot(x="country",y="balance",data=df1)

plt.xticks(rotation = 90)

plt.show()

sns.barplot(x="country",y="credit_score",data=df1)

plt.xticks(rotation = 90)

plt.show()

sns.lineplot(x="tenure",y="credit_score",data=df1,hue="balance",style="balance")

sns.scatterplot(x="estimated_value",y="product_number",hue="balance",data=df1)

sns.histplot(data=df1, x="credit_score", hue="balance", element="step", stat="density")

sns.relplot(data=df1,x=df1["credit_score"],y=df1["age"],hue="Species")

# OUTPUT:
# Data Cleaning Process:
![Screenshot (260)](https://github.com/Dhivya-bharathi88/ex.no.10-ds/assets/128019999/7567f0c1-9f94-4ad3-8af3-42ebb89744c8)
![Screenshot (261)](https://github.com/Dhivya-bharathi88/ex.no.10-ds/assets/128019999/4f0dd1af-4066-4222-92c7-6118e80bac33)
![Screenshot (262)](https://github.com/Dhivya-bharathi88/ex.no.10-ds/assets/128019999/39f49c93-c3e8-452c-93eb-9e015035e173)
![Screenshot (264)](https://github.com/Dhivya-bharathi88/ex.no.10-ds/assets/128019999/058e113a-0b70-4e25-8b96-2f830c82300d)

# Handling Outliers:

![Screenshot (265)](https://github.com/Dhivya-bharathi88/ex.no.10-ds/assets/128019999/269a01c3-dbdb-4e02-9924-e36ec9cf40b8)
![Screenshot (266)](https://github.com/Dhivya-bharathi88/ex.no.10-ds/assets/128019999/626e1557-8393-4241-bf2a-b8ec60ea0d61)
![Screenshot (267)](https://github.com/Dhivya-bharathi88/ex.no.10-ds/assets/128019999/c7edf1ba-bae3-4239-a331-a1e3a0f713c3)

# EDA Techniques:

![Screenshot (268)](https://github.com/Dhivya-bharathi88/ex.no.10-ds/assets/128019999/e8a8f2e5-25bb-4e01-bb1a-5ecc664a4c13)
![Screenshot (269)](https://github.com/Dhivya-bharathi88/ex.no.10-ds/assets/128019999/04e93481-9c3b-4959-96c9-380033037250)
![Screenshot (270)](https://github.com/Dhivya-bharathi88/ex.no.10-ds/assets/128019999/a7ed8058-581d-40a3-b45e-550b15f9dc51)
![Screenshot (271)](https://github.com/Dhivya-bharathi88/ex.no.10-ds/assets/128019999/91c541ac-44b3-44c0-a113-94d0d7b8a496)
![Screenshot (272)](https://github.com/Dhivya-bharathi88/ex.no.10-ds/assets/128019999/234ff663-45f6-4f3e-837f-e8a40beba292)
![Screenshot (273)](https://github.com/Dhivya-bharathi88/ex.no.10-ds/assets/128019999/0de0b52e-d50a-41ee-a281-752518e23c14)
![Screenshot (274)](https://github.com/Dhivya-bharathi88/ex.no.10-ds/assets/128019999/9a780f55-08de-4058-a059-2bc5537d3228)
![Screenshot (275)](https://github.com/Dhivya-bharathi88/ex.no.10-ds/assets/128019999/c55261d5-b489-4284-a587-c79aa52dcc05)
![Screenshot (276)](https://github.com/Dhivya-bharathi88/ex.no.10-ds/assets/128019999/7506b7d0-ff6c-44bf-b0eb-103a6904cd08)
![Screenshot (277)](https://github.com/Dhivya-bharathi88/ex.no.10-ds/assets/128019999/0ad17d6c-2215-415b-bb0c-5efa3100b399)

# Feature Transformation:

![Screenshot (278)](https://github.com/Dhivya-bharathi88/ex.no.10-ds/assets/128019999/0bab314c-3d25-437b-99bc-21567fd2ef03)
![Screenshot (279)](https://github.com/Dhivya-bharathi88/ex.no.10-ds/assets/128019999/054d94ea-697e-417f-bf06-f9624393f2b2)
![Screenshot (280)](https://github.com/Dhivya-bharathi88/ex.no.10-ds/assets/128019999/919b4193-b560-4804-a0e4-2324e16cf97a)
![Screenshot (281)](https://github.com/Dhivya-bharathi88/ex.no.10-ds/assets/128019999/b222fc59-5090-4a89-964d-70f0e5202da8)
![Screenshot (282)](https://github.com/Dhivya-bharathi88/ex.no.10-ds/assets/128019999/5f6f51ec-64d2-40ca-acad-163c4ddf594b)
![Screenshot (283)](https://github.com/Dhivya-bharathi88/ex.no.10-ds/assets/128019999/6957a9ff-3c01-4385-8345-2ceaf487225d)
![Screenshot (284)](https://github.com/Dhivya-bharathi88/ex.no.10-ds/assets/128019999/1361d410-155a-427e-9f8e-221c159a7c35)
![Screenshot (285)](https://github.com/Dhivya-bharathi88/ex.no.10-ds/assets/128019999/ec73bbbb-f35b-493f-bdab-ac35a949b523)
![Screenshot (286)](https://github.com/Dhivya-bharathi88/ex.no.10-ds/assets/128019999/33e82ce0-fcd6-463d-b20d-3b4a835b7c89)
# Data Visualisation:
![Screenshot (287)](https://github.com/Dhivya-bharathi88/ex.no.10-ds/assets/128019999/886dd30e-1317-450b-870b-393a855a006c)
![Screenshot (288)](https://github.com/Dhivya-bharathi88/ex.no.10-ds/assets/128019999/f6ef788d-f348-4627-8ee9-5bfe960fed96)

# RESULT:

Thus the Data Science Process on Complex Dataset were performed and output was verified successfully.
