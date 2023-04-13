#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install pyspark


# In[2]:


#pip install findspark


# In[ ]:





# ### _Using findspark for creating a bigde between python and spark._
# ### _Pointed the spark library in INIT()_

# In[3]:


import findspark
findspark.init('/opt/homebrew/Cellar/apache-spark/3.2.1/libexec/')


# ### _Using pyspark related libraries_

# In[4]:


from pyspark import SparkConf,SparkContext


# In[5]:


from pyspark.sql import SparkSession


# In[6]:


from pyspark.sql.functions import isnan, when, count, col


# In[7]:


from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator


# ### *Python libraries for handling the data and plots*

# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[9]:


sc = SparkSession.builder.master("local[*]").appName("boston_house_pricing").getOrCreate()


# In[10]:


sc


# ### *Reading csv file using pyspark methods*

# In[11]:


boston_housing_file_path = '/Users/jithesh_sunny/Desktop/CW_7082/housing.csv'


# In[12]:


boston_house_file_df = sc.read.csv(boston_housing_file_path,header=True,inferSchema=True)


# In[13]:


boston_house_file_df.printSchema()


# In[14]:


boston_house_file_df.toPandas()


# In[15]:


boston_house_file_df.select([count(when(isnan(c), c)).alias(c) for c in boston_house_file_df.columns]).show()


# In[16]:


boston_house_file_df.dropDuplicates().count()


# In[ ]:





# In[17]:


boston_house_file_df.toPandas().plot(kind='box', subplots=True, layout=(4,7), figsize=(15,15), sharex=False, sharey=False)
plt.show()


# In[18]:


for element in boston_house_file_df.columns[0:len(boston_house_file_df.columns)-1]:
    if not( isinstance(boston_house_file_df.select(element).take(1)[0][0],str)):
        print( "Correlation to MEDV for ", element, boston_house_file_df.stat.corr('MEDV',element))


# ### Correlation-coefficients

# In[19]:


boston_house_file_df.columns[0:len(boston_house_file_df.columns)-1]


# In[20]:


positive_corr_col = ["ZN", "CHAS", "RM", "DIS", "B",'RAD', ]
all_col = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']


# ### Generating VectorAssembler()

# In[21]:


vectorAssembler = VectorAssembler(inputCols = all_col , outputCol = 'features')
vect_boston_house_df = vectorAssembler.transform(boston_house_file_df)
vect_boston_house_df.show(5)


# ### Renaming the column MEDV to Label
# 

# In[22]:


vect_boston_house_df = vect_boston_house_df.select(['features', 'MEDV'])
vect_boston_house_df = vect_boston_house_df.withColumnRenamed("MEDV","label")
vect_boston_house_df.show(3)


# ### Creating the Train and Test data

# In[23]:


train_vect_house_data,test_vect_house_data = vect_boston_house_df.randomSplit([0.85, 0.15])


# In[24]:


train_vect_house_data.show()


# ### Generating Linear regression object and model using train data
# 

# In[25]:


lr_vectr_boston_house = LinearRegression(featuresCol = 'features', labelCol='label', 
                                         maxIter=10, regParam=0.15, elasticNetParam=0.85)
lr_vectr_boston_house_model = lr_vectr_boston_house.fit(train_vect_house_data)

print("Coefficients: " + str(lr_vectr_boston_house_model.coefficients))
print("Intercept: " + str(lr_vectr_boston_house_model.intercept))


# ### Summarizing the data from model 

# In[26]:


boston_house_train_summary = lr_vectr_boston_house_model.summary
print("RMSE: %f" % boston_house_train_summary.rootMeanSquaredError)
print("r2: %f" % boston_house_train_summary.r2)


# In[27]:


train_vect_house_data.describe().show()


# ### Passing the test data to the model

# In[28]:


lr_vectr_boston_house_predictions = lr_vectr_boston_house_model.transform(test_vect_house_data)


# In[29]:


lr_vectr_boston_house_evaluator = RegressionEvaluator(predictionCol="prediction",labelCol="label",metricName="r2")


# In[30]:


test_result = lr_vectr_boston_house_model.evaluate(test_vect_house_data)
print("Root Mean Squared Error (RMSE) on test data = %g" % test_result.rootMeanSquaredError)
print("R Squared (R2) on test data = ",lr_vectr_boston_house_evaluator.evaluate(lr_vectr_boston_house_predictions))


# In[31]:


lr_vectr_boston_house_predictions.show()


# In[ ]:





# In[ ]:




