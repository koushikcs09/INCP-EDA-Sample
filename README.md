




# *****************************************
# EDA_sample.md
 # *****************************************
# Read Input File (Excel) and provide Target Column


```python
import pandas as pd
df=pd.read_excel('Sample_input.xlsx',headers=0)

# Please provide the name of Target Column
target_nm='target'
```


```python
print(df)
```

       f1  f2 f3  f4  target
    0   a  10  x   0       1
    1   a  20  x   1       2
    2   b   1  y   0       5
    3   a  15  x   0       3
    4   b  30  y   0       0
    5   a  35  x   0       1
    6   a  40  y   1       1
    7   a  10  y   1       1
    8   b  10  y   1       1
    9   b  25  y   1       2
    10  b   0  x   0       3
    11  b  15  y   0       9
    12  b  30  y   1       1
    13  a  35  x   0       2
    14  b  40  y   0       7
    15  a  15  x   0       1
    16  b  30  y   1       1
    17  a  35  x   1       1
    18  b  40  x   1       5
    

# Target Variable Distribution


```python
c = df[target_nm].value_counts(dropna=False)
p = df[target_nm].value_counts(dropna=False, normalize=True)
pd.concat([c,p], axis=1, keys=['counts', '%']).to_excel("Target_Variable_Distribution.xlsx", header=True)
print(pd.concat([c,p], axis=1, keys=['counts', '%']))
```

       counts         %
    1       9  0.473684
    2       3  0.157895
    5       2  0.105263
    3       2  0.105263
    9       1  0.052632
    7       1  0.052632
    0       1  0.052632
    

# Split Continuous and Categorical features


```python
import numpy as np
df_cont = df.select_dtypes(include=[np.number])
df_cat = df.select_dtypes(exclude=[np.number]) 
print(df_cont.info())
print(df_cat.info())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 19 entries, 0 to 18
    Data columns (total 3 columns):
    f2        19 non-null int64
    f4        19 non-null int64
    target    19 non-null int64
    dtypes: int64(3)
    memory usage: 536.0 bytes
    None
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 19 entries, 0 to 18
    Data columns (total 2 columns):
    f1    19 non-null object
    f3    19 non-null object
    dtypes: object(2)
    memory usage: 384.0+ bytes
    None
    

# Continuous Feature Descriptive Stats


```python
from Descriptive_Stats import descriptive_analysis
descriptive_analysis(df,df_cont,target_var=target_nm,out_file="Descriptive_Stats_Continuous.xlsx")
```

      variable     n  nmiss  min   max       mean  median    p1   p5  p10   p25  \
    3       f4  19.0      0  0.0   1.0   0.473684     0.0  0.00  0.0  0.0   0.0   
    1       f2  19.0      0  0.0  40.0  22.947368    25.0  0.18  0.9  8.2  12.5   
    
        p75   p90   p95   p99      corr  
    3   1.0   1.0   1.0   1.0 -0.329244  
    1  35.0  40.0  40.0  40.0 -0.067499  
    

# Continuous Feature Rank & Plot


```python
from Continuous_Field_Rank_Plot_Regression import *
cont_bin(df,df_cont,n_bin=10,target_col=target_nm,filename="Continuous_Base.xlsx")        
add_table_plot(df_cont,in_file="Continuous_Base.xlsx",sheet_nm='Continous',target_col=target_nm,out_file="Continuous_Rank_Plot.xlsx")
```

# Categorical Feature Rank & Plot


```python
from Categorical_Field_Rank_Plot_Regression import *
cat_bin(df,df_cat,target_col=target_nm,filename="Categorical_Base.xlsx")       
add_table_plot(df,in_file= "Categorical_Base.xlsx",sheet_nm='Categorical',target_col=target_nm,out_file="Categorical_Rank_Plot.xlsx")
```

# Continuous Feature WOE,IV


```python
from Continuous_Field_WOE_IV_Regression import *
cont_bin(df,df_cont,n_bin=10,target_col=target_nm,filename="Continuous_WOE.xlsx")
```

      Variable  IV_value
    0       f2  0.279511
    1       f4  0.100807
    

# Categorical Feature WOE,IV


```python
from Categorical_Field_WOE_IV_Regression import * 
cat_bin(df,df_cat,target_col=target_nm,filename="Categorical_WOE.xlsx")
df=automate_woe_population(df,df_cat,filename="Categorical_WOE.xlsx")
```

    df['f1_WOE']=0
    df.loc[df['f1']=="b",'f1_WOE']=0.3180668090784977
    df.loc[df['f1']=="a",'f1_WOE']=-0.5379838424183007
    df['f3_WOE']=0
    df.loc[df['f3']=="y",'f3_WOE']=0.1239107946375402
    df.loc[df['f3']=="x",'f3_WOE']=-0.158494220713397
    

 # *****************************************
# HyperTuning_Grid_Search_Format.md
 # *****************************************
```python
import pandas as pd 

def format_grid_search_result(res):
    ''' This User Defined Function formats the Grid Search results for better readability '''
    
    global df_gs_result
    gs_results=res
    
    gs_model=gs_results['params']
    
    # Grid Search : AUC Metrics
    gs_mean_test_AUC=pd.Series(gs_results['mean_test_AUC'])
    gs_std_test_AUC=pd.Series(gs_results['std_test_AUC'])
    gs_rank_test_AUC=pd.Series(gs_results['rank_test_AUC'])
    
    # Grid Search : Accuracy Metrics
    gs_mean_test_Accuracy=pd.Series(gs_results['mean_test_Accuracy'])
    gs_std_test_Accuracy=pd.Series(gs_results['std_test_Accuracy'])
    gs_rank_test_Accuracy=pd.Series(gs_results['rank_test_Accuracy'])
    
    # Grid Search : Recall Metrics
    gs_mean_test_Recall=pd.Series(gs_results['mean_test_Recall'])
    gs_std_test_Recall=pd.Series(gs_results['std_test_Recall'])
    gs_rank_test_Recall=pd.Series(gs_results['rank_test_Recall'])

    # Grid Search : Precision Metrics
    gs_mean_test_Precision=pd.Series(gs_results['mean_test_Precision'])
    gs_std_test_Precision=pd.Series(gs_results['std_test_Precision'])
    gs_rank_test_Precision=pd.Series(gs_results['rank_test_Precision'])
    
    # Grid Search : F1-Score Metrics
    gs_mean_test_F1_Score=pd.Series(gs_results['mean_test_F1 Score'])
    gs_std_test_F1_Score=pd.Series(gs_results['std_test_F1 Score'])
    gs_rank_test_F1_Score=pd.Series(gs_results['rank_test_F1 Score'])   

    
    gs_model_split=str(gs_model).replace("[{","").replace("}]","").split('}, {')
    df_gs_result=pd.DataFrame(gs_model_split,index=None,columns=['Model_attributes'])
    df_gs_result=pd.concat([df_gs_result,gs_mean_test_AUC,gs_std_test_AUC,gs_rank_test_AUC,gs_mean_test_Accuracy,gs_std_test_Accuracy,gs_rank_test_Accuracy,gs_mean_test_Recall,gs_std_test_Recall,gs_rank_test_Recall,gs_mean_test_Precision,gs_std_test_Precision,gs_rank_test_Precision,gs_mean_test_F1_Score,gs_std_test_F1_Score,gs_rank_test_F1_Score],axis=1)
    
    df_gs_result.columns=['Model_attributes','mean_test_AUC','std_test_AUC','rank_test_AUC','mean_test_Accuracy','std_test_Accuracy','rank_test_Accuracy','mean_test_Recall','std_test_Recall','rank_test_Recall','mean_test_Precision','std_test_Precision','rank_test_Precision','mean_test_F1_Score','std_test_F1_Score','rank_test_F1_Score']
    
    return(df_gs_result) 


```

