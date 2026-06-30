from statistics import linear_regression
import pandas as pd
import numpy as np
from numpy.ma.extras import hstack
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error ,r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sqlalchemy import create_engine
from urllib.parse import quote_plus
import scipy.sparse as sp
import pickle
import time



# This part connects the model to the database
# 1. Connecting to the database using SQLAlchemy

db_password = quote_plus("YOUR_DATABASE_PASSWORD")
engine = create_engine(f"mysql+mysqlconnector://root:"
                       f"{db_password}@127.0.0.1/Project_Database")

print("loading Data base ....")

# 2. SQL query to join dimension tables with the fact table and retrieve the required columns.
query = """
    SELECT 
        j.Job_Title,
        j.Short_Description,
        j.Full_Job_Text,
        j.Category,
        l.Country_Code AS Location,
        f.Hourly_Rate_Avg
    FROM Fact_Job_Posting f
    JOIN Dim_Job j ON f.Job_Key = j.Job_Key
    JOIN Dim_Location l ON f.Location_Key = l.Location_Key
"""

# Reading data directly and converting it into a DataFrame.
Data = pd.read_sql(query, engine)

print(f"Successfully loaded {len(Data)} rows from the database!")

# 3. Merging text (handling NaN values ​​to ensure no errors occur)
Data["all_text"] = Data["Job_Title"].fillna("") + " " + Data["Short_Description"].fillna("") + " " + Data["Full_Job_Text"].fillna("")

# 4. Defining the target directly from the fact table.
y = Data["Hourly_Rate_Avg"]

# 1. Convert text into "words" rather than "complete sentences."
# We will take the 500 most frequently occurring words in the description.
tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
x_text_tfidf = tfidf.fit_transform(Data['all_text'])

# It converts to 0s and 1s.
x_wait = pd.get_dummies(Data[['Category', 'Location']], drop_first=True)
dummy_columns = x_wait.columns.tolist()

x= sp.hstack((x_text_tfidf,x_wait ))

# We calculate the average of the highest and lowest hours.
# y= (Data["Hourly Rate (Lower)"]+ Data["Hourly Rate (Upper)"]) / 2




# split data set
#80%>>>Train , 20%>>test
x_train,x_test, y_train,y_test = train_test_split(x,y , test_size=0.2 ,random_state=42)

#Bulid model
# lin_reg =linear_regression()
lin_reg = Ridge(alpha=0.01)

#Train model
lin_reg.fit(x_train,y_train)

#predicit

y_pred=lin_reg.predict(x_test)

y_pred = np.clip(y_pred , 15.0 , 150.0)


#Evaluate
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test ,y_pred)

print("Evaluation Metrics >>>>")
print("mse :" ,mse)
print("r2 :", r2)

print("-"*150)


# 6. Saving important files for the UI.
with open("salary_model.pkl", "wb") as f:
    pickle.dump({
        "model": lin_reg,
        "tfidf": tfidf,
        "dummy_columns": dummy_columns,
        "categories": Data['Category'].dropna().unique().tolist(),
        "locations": Data['Location'].dropna().unique().tolist()
    }, f)

print("Model and components saved successfully!")










































