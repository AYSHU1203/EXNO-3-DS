## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
            import pandas as pd

            df=pd.read_csv("C:\\Users\\admin\\Downloads\\data.csv")
            df

<img width="1046" height="591" alt="Screenshot 2025-09-24 113648" src="https://github.com/user-attachments/assets/b4d4bf05-dd8f-4319-8e5b-cda7ba1f2b13" />

       from sklearn.preprocessing import OrdinalEncoder,LabelEncoder

      df1=df.copy()
      df1=df.copy()
      education=["High School","Diploma","Bachelors","Masters","PhD"]
      enc=OrdinalEncoder(categories=[education])
      enc.fit_transform(df1[['Ord_2']])
     df1['ordinalencoder']=enc.fit_transform(df1[['Ord_2']])
     df1
     
 <img width="1139" height="585" alt="Screenshot 2025-09-24 113811" src="https://github.com/user-attachments/assets/db0ce6f0-7064-4b6d-9137-237f1cb86300" />

      df2=df.copy()
      enc=LabelEncoder()
      df1['LabelEncoder']=enc.fit_transform(df1[['Ord_2']])
      df1

<img width="1211" height="586" alt="Screenshot 2025-09-24 113917" src="https://github.com/user-attachments/assets/2726e76d-9df9-4a35-b406-73c0934a7eb7" />

        from sklearn.preprocessing import OneHotEncoder

        df3=df.copy()
        enc=OneHotEncoder()
        newdata=pd.DataFrame(enc.fit_transform(df3[['City']]))
        df4=pd.concat([df3,newdata],axis=1)
        df4

<img width="1134" height="564" alt="Screenshot 2025-09-24 114048" src="https://github.com/user-attachments/assets/562eff03-c83e-47e6-8f69-a862122941c4" />

         pd.get_dummies(df4,columns=['City'])

  
<img width="1232" height="485" alt="Screenshot 2025-09-24 114131" src="https://github.com/user-attachments/assets/80a03415-505f-4453-b8dc-6f54c77d55e8" />

          pip install --upgrade category_encoders


<img width="1219" height="492" alt="Screenshot 2025-09-24 114223" src="https://github.com/user-attachments/assets/1d7c0b0c-ca77-4ca2-b271-79e7ab277e9b" />

           from category_encoders import BinaryEncoder

           df5=df.copy()
           enc=BinaryEncoder()
           newdata=pd.DataFrame(enc.fit_transform(df5[['Ord_1']]))
           df6=pd.concat([df5,newdata],axis=1)
           df6

  
<img width="1181" height="558" alt="Screenshot 2025-09-24 114332" src="https://github.com/user-attachments/assets/7724cd7c-865f-4d44-9b82-a95c3fb0456b" />

            from category_encoders import TargetEncoder

            df7=df.copy()
            enc=TargetEncoder()
            newdata=pd.DataFrame(enc.fit_transform(df7[['Ord_1']],df7['Target']))
            df8=pd.concat([df7,newdata],axis=1)
            df8

<img width="1049" height="552" alt="Screenshot 2025-09-24 114426" src="https://github.com/user-attachments/assets/98e2e1e8-faa2-4ff2-827f-213c3e1c3d65" />

## 2.DATA_TRANSFORM

        import pandas as pd
        df=pd.read_csv("C:\\Users\\admin\\Downloads\\Data_to_Transform.csv")
        df

<img width="1210" height="700" alt="Screenshot 2025-09-24 114550" src="https://github.com/user-attachments/assets/134436e0-e5f1-40c4-ac35-203d86f4ef5d" />

        df.skew()

<img width="558" height="178" alt="Screenshot 2025-09-24 114649" src="https://github.com/user-attachments/assets/29242b9c-7e41-4d11-a713-840a95ca410a" />

        import numpy as np
        df1=df.copy()
        df['log transformation']=np.log(df["Moderate Positive Skew"])
        df1

<img width="1223" height="679" alt="Screenshot 2025-09-24 114804" src="https://github.com/user-attachments/assets/c90f1242-b476-419d-8993-2d8494eceaae" />

       import statsmodels.api as sm
       import matplotlib.pyplot as plt


<img width="1216" height="784" alt="Screenshot 2025-09-24 114840" src="https://github.com/user-attachments/assets/ad984445-afdf-4aea-90ee-ab12b021a347" />

       sm.qqplot(df["Highly Positive Skew"],line="45")
       plt.show()

<img width="1168" height="739" alt="Screenshot 2025-09-24 114916" src="https://github.com/user-attachments/assets/aae43840-4207-40f0-908e-4b2efede9fc5" />

       sm.qqplot(df["Moderate Negative Skew"],line="45")
       plt.show()

<img width="1183" height="726" alt="Screenshot 2025-09-24 115005" src="https://github.com/user-attachments/assets/202eacb0-5106-461c-a72b-920a666acade" />

       sm.qqplot(df["Highly Negative Skew"],line="45")
       plt.show()

<img width="1197" height="831" alt="Screenshot 2025-09-24 115103" src="https://github.com/user-attachments/assets/9af7e25f-0264-4003-9d8c-b9d1cae6bbdc" />

       sm.qqplot(df["log transformation"],line="45")
       plt.show()

<img width="1189" height="749" alt="Screenshot 2025-09-24 115148" src="https://github.com/user-attachments/assets/fd8e9cf6-54e6-40da-a0c0-92b743b93135" />

       df2=df.copy()
       df2["sqrt transformation"]=np.sqrt(df["Moderate Positive Skew"])
       df2


<img width="1238" height="533" alt="Screenshot 2025-09-24 115231" src="https://github.com/user-attachments/assets/3c8fd8a8-7214-4d6b-bef2-c5ac7c1f2d92" />

      sm.qqplot(df["sqrt transformation"],line="45")
      plt.show()


<img width="1204" height="781" alt="Screenshot 2025-09-24 115315" src="https://github.com/user-attachments/assets/03996611-3091-4132-b8b5-46792153b3a1" />

      df3=df.copy()
      df3['square transformation']=np.square(df["Moderate Positive Skew"])
      df3

<img width="1198" height="497" alt="Screenshot 2025-09-24 115354" src="https://github.com/user-attachments/assets/3d88840b-5f9f-42d2-b841-ebae35593d83" />

     sm.qqplot(df3["square transformation"],line="45")
     plt.show()

<img width="1205" height="792" alt="Screenshot 2025-09-24 115443" src="https://github.com/user-attachments/assets/687beb3f-cf81-4be8-8e2d-327624b41376" />

     df4=df.copy()
     df4['reciprocal transformation']=np.square(df["Moderate Positive Skew"])
     df4

<img width="1215" height="471" alt="Screenshot 2025-09-24 115524" src="https://github.com/user-attachments/assets/f96185fb-8885-4ae9-a725-22397d8cfca1" />

    sm.qqplot(df4["reciprocal transformation"],line="45")
    plt.show()

<img width="1119" height="710" alt="Screenshot 2025-09-24 115600" src="https://github.com/user-attachments/assets/1b9dd403-b908-4180-b48f-7c72bd06e18f" />

    from scipy import stats

     df5=df.copy()
     df['boxcox transformation'],p=stats.boxcox(df5["Moderate Positive Skew"])
     df5

<img width="1246" height="525" alt="Screenshot 2025-09-24 115655" src="https://github.com/user-attachments/assets/24dca295-111a-4ede-8554-792d248def55" />

     sm.qqplot(df["boxcox transformation"],line="45")
     plt.show()

<img width="1121" height="728" alt="Screenshot 2025-09-24 115822" src="https://github.com/user-attachments/assets/32817716-e636-4d9d-b408-05a0c843aa94" />

    df6=df.copy()
    df6['yeojohnson transformation'],p=stats.boxcox(df5["Moderate Positive Skew"])
    df6

<img width="1198" height="467" alt="Screenshot 2025-09-24 115912" src="https://github.com/user-attachments/assets/70ac7a03-bdfa-4554-9ed5-9f1ea8c73d82" />

    sm.qqplot(df6["yeojohnson transformation"],line="45")
    plt.show()

<img width="1142" height="800" alt="Screenshot 2025-09-24 120023" src="https://github.com/user-attachments/assets/400053c3-50bc-44a7-83b1-a466fbbb8f18" />


    from sklearn.preprocessing import QuantileTransformer

    df7=df.copy()
    qt=QuantileTransformer(output_distribution="normal")
    df7['QuantileTransformation']=qt.fit_transform(df7[["Highly Positive Skew"]])
    df7

<img width="1198" height="475" alt="Screenshot 2025-09-24 120105" src="https://github.com/user-attachments/assets/a093f2a5-ed2e-458b-acca-91833cc24d35" />

    sm.qqplot(df7['QuantileTransformation'],line="45")
    plt.show()

<img width="1134" height="816" alt="Screenshot 2025-09-24 120148" src="https://github.com/user-attachments/assets/e6a78eb7-ee9d-47ee-91ec-abef61cfc486" />

            
# RESULT:
        Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.

       
