
# *****************************************
# EDA_sample
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
# *****************************************
# Descriptive_Stats.py
 # *****************************************
```python
import pandas as pd

def descriptive_analysis(df,df_continuous,target_var,out_file):
    ''' This User Defined Function creates descriptive statistics of continuous variables 
    1. Univariate Statistics (Count,Mean,Quartile,Maximum,Minimum)
    2. Missing Count
    3. Correlation with Target'''
    
    # Transposing the matrix returned by describe() <Descriptive statistics like count,mean,standrand deviation> ,so that the attributes of the dataset are coming as row-wise for better readability. <Note : This is having only conitnous attributes>
    df_univariate = df.describe().T

    # Checking the missing count for each attributes of the dataset <Note : This is having both conitnous and categorical attributes>
    df_nmis = df.isnull().sum()

    # Resetting the index to the column names 
    df_univariate = df_univariate.reset_index().set_index('index', drop=False)
    df_nmis = df_nmis.reset_index().set_index('index', drop=False)

    # Add the Missing Count information for each attribute
    df_univariate_2 = pd.concat([df_nmis,df_univariate], axis=1, join_axes=[df_nmis.index], join = 'outer')

    # Renaming the Missing Count column name as "nmiss"
    df_univariate_2 = df_univariate_2.rename(columns={ df_univariate_2.columns[1]: "nmiss" })

    # Dropping the Redundant columns.The output is having continous column with univariate statistics and their missing count ,while catgeorical columns are having therir only missing count and rest univariate statistics as "NAN"
    df_univariate_2 = df_univariate_2.drop([df_univariate_2.columns[0], df_univariate_2.columns[2]], axis=1)

    df_percentile = df_continuous.quantile([.01, .05, .1, .9, .95, .99]).T
    df_percentile = df_percentile.reset_index().set_index('index', drop=False)
    df_univariate_3 = pd.concat([df_univariate_2,df_percentile], axis=1, join_axes=[df_univariate_2.index], join = 'outer')
    df_univariate_3 = df_univariate_3.rename(columns={ df_univariate_3.columns[len(df_univariate_3.keys())-1]: "p99" , df_univariate_3.columns[len(df_univariate_3.keys())-2]: "p95", df_univariate_3.columns[len(df_univariate_3.keys())-3]: "p90", df_univariate_3.columns[len(df_univariate_3.keys())-4]: "p10", df_univariate_3.columns[len(df_univariate_3.keys())-5]: "p5", df_univariate_3.columns[len(df_univariate_3.keys())-6]: "p1"})
    df_univariate_3 = df_univariate_3.drop([df_univariate_3.columns[len(df_univariate_3.keys())-7]], axis=1)
    df_continuous_corr = pd.DataFrame(df_continuous.drop(target_var, axis=1).apply(lambda x: x.corr(df_continuous.eval(target_var))))
    df_continuous_corr = df_continuous_corr.reset_index().set_index('index', drop=False)
    df_continuous_corr.rename(columns={df_continuous_corr.columns[len(df_continuous_corr.keys())-1]: "corr"}, inplace=True)
    df_univariate_4 = pd.concat([df_univariate_3,df_continuous_corr], axis=1, join_axes=[df_univariate_3.index], join = 'outer')
    df_univariate_4 = df_univariate_4.drop([df_univariate_4.columns[len(df_univariate_4.keys())-2]], axis=1)

    df_continuous_column = list(df_continuous.columns)
    df_continuous_column = pd.Series(df_continuous_column)
    df_univariate_4 = df_univariate_4.reset_index()
    df_univariate_4 = df_univariate_4.rename(columns={'index':'variable','count':'n','50%':'median','25%':'p25','75%':'p75'})
    df_univariate_4 = df_univariate_4[['variable','n','nmiss','min','max','mean','median','p1','p5','p10','p25','p75','p90','p95','p99','corr']]
    df_univariate_4.sort_values(by=['variable'], ascending=False).dropna().to_excel(out_file, index = None, header=True)    
    print(df_univariate_4.sort_values(by=['variable'], ascending=False).dropna())
    

```
   

# Continuous Feature Rank & Plot


```python
from Continuous_Field_Rank_Plot_Regression import *
cont_bin(df,df_cont,n_bin=10,target_col=target_nm,filename="Continuous_Base.xlsx")        
add_table_plot(df_cont,in_file="Continuous_Base.xlsx",sheet_nm='Continous',target_col=target_nm,out_file="Continuous_Rank_Plot.xlsx")
```
 
# *****************************************
# Continuous_Field_Rank_Plot_Regression.py
# *****************************************
```python
```################ Continuous Field Rank and Plot for Regression Model #################```
import xlsxwriter
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mimg
import numpy as np
import os
import math
import pandas as pd
import numpy as np

''' Creating two new directory in the working directory to store the Plots of variables ''' 
work_dir=os.getcwd()
print(work_dir)
new_dir=work_dir+"/line_plot/"
check_dir_present=os.path.isdir(new_dir)
if check_dir_present == False:
    os.mkdir(new_dir)
    print("New Directory created : " + str(new_dir))
else:
    print("Existing Directory used : " + str(new_dir))
    
new_dir=work_dir+"/scatter_plot/"
check_dir_present=os.path.isdir(new_dir)
if check_dir_present == False:
    os.mkdir(new_dir)
    print("New Directory created : " + str(new_dir))
else:
    print("Existing Directory used : " + str(new_dir))

def plot_stat(df_continuous,title_name,target_col):
    '''This User Defined Functions performs following
    1. Line Plot showing Volume% and Event% against 'BINS' of the Continuous Variables
    2. Scatter Plot showing Target against 'BINS' of the Continuous Variables'''
    
    global df_plot,file_i,img_name,work_dir,scatter_img_name
    
    ''' Line Plot '''
    width=0.35
    df_plot['Volume(%)']=df_plot['Volume(%)']*100
    df_plot['Avg_Call_Volume']=df_plot['Avg_Call_Volume']

    df_plot.plot(x='BINS',y='Volume(%)',kind='bar',width=width,label=('Volume(%)'),color='b')
    plt.ylabel('Volume(%)',color='b')
    plt.ylim((0,100))

    plt.legend(loc='upper right')

    df_plot['Avg_Call_Volume'].plot(secondary_y=True,label=('Avg_Call_Volume'),color='r')
    plt.ylabel('Avg_Call_Volume',color='r')
    plt.ylim((0,100))
    plt.legend(loc='upper right')
    
    axis_1=plt.gca()

    for i,j in zip(df_plot['Volume(%)'].index,df_plot['Volume(%)']):
        i=round(i,2)
        j=round(j,2)
        axis_1.annotate('%s' %j,xy=(i,j),color='k')
        axis_1.annotate('%s' %i,xy=(i,j),color='k')
        
    for i,j in zip(df_plot['Avg_Call_Volume'].index,df_plot['Avg_Call_Volume']):
        i=round(i,2)
        j=round(j,2)
        axis_1.annotate('%s' %j,xy=(i,j),color='k')
        axis_1.annotate('%s' %i,xy=(i,j),color='k')

    plt.xlim([-width,len(df_plot['Volume(%)'])-width])
    plt.title(title_name)
    plt.xlabel("BINS")
    plt.grid()
    img_name=str(work_dir)+str("/line_plot/")+str(title_name)+ str(".png")
    plt.savefig(img_name,dpi=300,bbox_inches='tight')
    plt.clf()
    
    ''' Scatter Plot '''
    fig=plt.figure(figsize=(6,4))
    plt.scatter(df_continuous[title_name],df_continuous[target_col],c='DarkBlue')
    plt.ylabel(target_col,color='b')
    plt.xlabel(title_name,color='b')
    plt.title(title_name)
    plt.grid()
    scatter_img_name=str(work_dir)+str("/scatter_plot/")+str(title_name)+ str(".png")
    fig.savefig(scatter_img_name,dpi=300,bbox_inches='tight')
    plt.clf()

def add_table_plot(df_continuous,in_file,sheet_nm,out_file,target_col):
    ''' This User Defined Function adds Line Plot and Scatter Plot in an excel 
    just on the beside of the binning data of the categorical Fields.
    This gives better readability in analysing the data.
    '''
    
    global df_plot,img_name,work_dir,scatter_img_name
    
    work_dir=os.getcwd()
    df_cont=pd.read_excel(in_file,header=None,sheet_name=sheet_nm)
    df_cont.columns=['BINS','MIN','MAX','Range','Avg_Call_Volume','TOTAL','Volume(%)']
    df_cont=df_cont.fillna('')
    wb=xlsxwriter.Workbook(out_file)
    ws=wb.add_worksheet(sheet_nm)
    
    for i in range(len(df_cont)):
        
        for j in range(len(df_cont.columns)):
            if j == 0:
                col_pos= 'A'
            if j == 1:
                col_pos= 'B'
            if j == 2:
                col_pos= 'C'  
            if j == 3:
                col_pos= 'D'
            if j == 4:
                col_pos= 'E'
            if j == 5:
                col_pos= 'F'
            if j == 6:
                col_pos= 'G'  
            if j == 7:
                col_pos= 'H'
            if j == 8:
                col_pos= 'I'

            wrt_pos=str(col_pos)+str(i)
            ws.write(wrt_pos,df_cont.iloc[i,j])

        if df_cont.iloc[:,0][i] == "BINS":
            pos_min_loc=i+int(1)
            if pos_min_loc==1:
                title_name=df_cont.columns[0]
            else:
                title_loc=int(pos_min_loc)-int(2)
                title_name=df_cont.iloc[:,0][title_loc]
        
        if df_cont.iloc[:,0][i] == "Total":
            pos_max_loc=i
            df_plot=df_cont[pos_min_loc:pos_max_loc]
            df_plot.columns=['BINS','MIN','MAX','Range','Avg_Call_Volume','TOTAL','Volume(%)']
            df_plot=df_plot.reset_index()
            
            ''' Calls plot_stat() to create line plot and scatter plot '''
            plot_stat(df_continuous,title_name,target_col)
            img_pos=str('K')+ str(pos_min_loc-int(2))   
            img2=mimg.imread(img_name)
            imgplot2=plt.imshow(img2)

            ''' Inset line plot in the excel '''
            ws.insert_image(img_pos,img_name,{'x_scale' : 0.6, 'y_scale' : 0.6})
            
            ''' Insert Scatter plot in the excel '''
            scatter_img_pos=str('Q')+ str(pos_min_loc-int(2))  
            ws.insert_image(scatter_img_pos,scatter_img_name,{'x_scale' : 0.6, 'y_scale' : 0.6})
    wb.close()

def create_volume_group(df_continuous,curr_var,target_col,n_bin):
    '''This User defined function creates bins on the basis of parameter n_bin (number of bins) 
    provided
    This algorithm creates almost eqi_volume groups with unique values in groups :  
    1. It calculates the Average Bin Volume by dividing the total volume of data by number of 
    bins
    2. It sorts the data based on the value of the continous variables
    3. It directly moves to index (I1) having value Average Bin Volume (ABV) 
    4. It checks the value of continous variable at the index position decided in previosu step
    5. It finds the index(I2) of last position of the value identified at previous step 4
    6. It concludes the data of the First Bin within the range (0 to I2)
    7. The Index I1 is again calculated as I1 = I2 + ABV and step 4-6 is repeated
    8. This Process is continued till the desired number of bins are created
    9. Seperate Bin is created if the continuous variable is having missing value
    
    Note : qcut() does provide equi-volume groups but does not provide unique values in the
    groups.
    hence,qcut() is not used in binning the data.
    '''
        
    global df_fnl
    df_continuous_column=[]
    df_continuous_column.append(curr_var)
    ttl_vol=len(df_continuous)
    avg_bin_vol = round(ttl_vol/n_bin)
    lst=[]
    df_fnl=pd.DataFrame(lst)
    df_mod=pd.DataFrame(lst)
    df_mod_null=pd.DataFrame(lst)

    for i in range(len(df_continuous_column)):
        if (df_continuous_column[i] != target_col):
            curr_var=df_continuous_column[i]

            df_mod1=pd.DataFrame(df_continuous[[curr_var,target_col]])

            #### Sort the Data ####
            df_mod1=df_mod1.sort_values(by=[curr_var],ascending=True)
            df_mod1=df_mod1.reset_index()
        
            df_mod_null=df_mod1[pd.isnull(df_mod1[curr_var])]
            df_mod1=df_mod1.dropna(subset=[curr_var]) 
        
            seq=list(range(1,len(df_mod1)+int(1)))
            df_seq=pd.DataFrame(seq,columns=['sequence'])

            df_mod1=pd.concat([df_mod1,df_seq],axis=1)
            
            '''Creating the Missing BIN'''
            if len(df_mod_null) > int(0):
                ttl_vol=len(df_mod1)
                avg_bin_vol = round(ttl_vol/n_bin)
                group_num='missing'
        
            pos_indx_max = 0
            val_list=df_mod1[curr_var].unique()
            
            '''Checks if the Unique Values of the continuos variables is 2,then simply create 2 bins out of two unique values'''
            if len(val_list) == 2:
                for i in range(len(val_list)):
                    val_of_indx=val_list[i]
                    df_mod3=df_mod1[df_mod1[curr_var] == val_of_indx]
                    group_num=i+int(1)
                    df_mod3['Decile']=group_num
                    df_fnl=pd.concat([df_fnl,df_mod3])
                    
            '''Checks if the Unique Values of the continuos variables is more than 2'''
            if len(val_list) != 2:        
                for bin_num in range(n_bin):
                    if  pos_indx_max < ttl_vol-1 :
                        '''For the First Group, index is assigned to length of average bin 
                        volume'''
                        if bin_num == 0:
                            indx = (bin_num+int(1)) * avg_bin_vol
                        else:
                            indx = int(pos_indx_max) + int(avg_bin_vol)
                        
                        '''Setting the Index for last Group'''
                        if indx > ttl_vol:
                            indx = ttl_vol-int(1)

                        val_of_indx = df_mod1[curr_var].iat[indx]
                
                        if math.isnan(val_of_indx) == True:
                            pos_indx_min=pos_indx_max+int(1)
                            pos_indx_max=ttl_vol-int(1)
                        else:
                            df_mod3=df_mod1[df_mod1[curr_var] == val_of_indx]
                            pos_indx_min = pos_indx_max
                            pos_indx_max = df_mod3['sequence'].iat[-1]
                       
                        df_mod3=df_mod1[pos_indx_min:pos_indx_max]
                        group_num=bin_num+int(1)
                        df_mod3['Decile']=group_num
                    
                        df_fnl=pd.concat([df_fnl,df_mod3])
            
            df_fnl=pd.concat([df_fnl,df_mod_null])
            

def cont_bin_Miss(df_dcl_fnl,df_continuous_column,i,target_col,filename):
    '''This User Defined Function creates BINS for continous variables having some Missing 
    values'''
    
    global continuous_target_nx

    '''The Missing Values have been grouped together as a seperate bin - "Missing" '''
    df_dcl_fnl[df_continuous_column[i]] = df_dcl_fnl[df_continuous_column[i]].astype(object)
    df_dcl_fnl[df_continuous_column[i]] = df_dcl_fnl[df_continuous_column[i]].fillna('Missing')
    df_dcl_fnl['Decile'] = df_dcl_fnl['Decile'].fillna('Missing')
    
    continuous_target_nx = pd.DataFrame(
        [df_dcl_fnl.groupby('Decile')[df_continuous_column[i]].min(),
         df_dcl_fnl.groupby('Decile')[df_continuous_column[i]].max(),
         df_dcl_fnl.groupby('Decile')[target_col].mean(),
         df_dcl_fnl.groupby('Decile')[df_continuous_column[i]].count()]
        ).T
    continuous_target_nx.columns = ["MIN","MAX","Avg_Call_Volume","TOTAL"]
    
    continuous_target_nx=continuous_target_nx.reset_index()
    list1=[]
    list_vol_pct=[]
    list_event_pct=[]

    for i in range(len(continuous_target_nx.MIN)):
        list1.append(str(continuous_target_nx['MIN'][i])+'-'+str(continuous_target_nx['MAX'][i]))
        list_vol_pct.append(continuous_target_nx['TOTAL'][i]/continuous_target_nx['TOTAL'].sum())

    continuous_target_nx = pd.concat([pd.Series(list1),continuous_target_nx,pd.Series(list_vol_pct)],axis=1)

    continuous_target_nx = continuous_target_nx[["Decile","MIN","MAX",0,"Avg_Call_Volume","TOTAL",1]]

    continuous_target_nx = continuous_target_nx.rename(columns={continuous_target_nx.columns[len(continuous_target_nx.keys())-4]: "Range"})
    continuous_target_nx = continuous_target_nx.rename(columns={continuous_target_nx.columns[len(continuous_target_nx.keys())-1]: "Volume(%)"})
    continuous_target_nx = continuous_target_nx.rename(columns={continuous_target_nx.columns[len(continuous_target_nx.keys())-7]: "BINS"})       

    continuous_target_nx = continuous_target_nx.append({"BINS":"Total",
                                 "MIN":" ",
                                 "MAX":" ",
                                 "Range":" ",
                                 "TOTAL":continuous_target_nx['TOTAL'].sum(),
                                 "Volume(%)":continuous_target_nx['Volume(%)'].sum(),
                                 "Avg_Call_Volume":df_dcl_fnl[target_col].mean()                                                  },ignore_index=True)
                                      
    
def cont_bin_NO_Miss(df_dcl_fnl,df_continuous_column,i,target_col,filename):
    '''This User defined function creates BINS for continuous variables having no missing
    values'''
    
    global continuous_target_nx

    continuous_target_nx = pd.DataFrame(
        [df_dcl_fnl.groupby('Decile')[df_continuous_column[i]].min(),
         df_dcl_fnl.groupby('Decile')[df_continuous_column[i]].max(),
         df_dcl_fnl.groupby('Decile')[target_col].mean(),
         df_dcl_fnl.groupby('Decile')[df_continuous_column[i]].count()]
        ).T
    continuous_target_nx.columns = ["MIN","MAX","Avg_Call_Volume","TOTAL"]

    continuous_target_nx=continuous_target_nx.reset_index()
    list1=[]
    list_vol_pct=[]
    list_event_pct=[]
    
    for i in range(len(continuous_target_nx.MIN)):
        list1.append(str(continuous_target_nx['MIN'][i])+'-'+str(continuous_target_nx['MAX'][i]))
        list_vol_pct.append(continuous_target_nx['TOTAL'][i]/continuous_target_nx['TOTAL'].sum())

    continuous_target_nx = pd.concat([pd.Series(list1),continuous_target_nx,pd.Series(list_vol_pct)],axis=1)

    continuous_target_nx = continuous_target_nx[["Decile","MIN","MAX",0,"Avg_Call_Volume","TOTAL",1]]

    continuous_target_nx = continuous_target_nx.rename(columns={continuous_target_nx.columns[len(continuous_target_nx.keys())-4]: "Range"})
    continuous_target_nx = continuous_target_nx.rename(columns={continuous_target_nx.columns[len(continuous_target_nx.keys())-1]: "Volume(%)"})
    continuous_target_nx = continuous_target_nx.rename(columns={continuous_target_nx.columns[len(continuous_target_nx.keys())-7]: "BINS"})       

    continuous_target_nx = continuous_target_nx.append({"BINS":"Total",
                                 "MIN":" ",
                                 "MAX":" ",
                                 "Range":" ",
                                 "TOTAL":continuous_target_nx['TOTAL'].sum(),
                                 "Volume(%)":continuous_target_nx['Volume(%)'].sum(),
                                 "Avg_Call_Volume":df_dcl_fnl[target_col].mean()
                                 },ignore_index=True)
    
    
def cont_bin(df,df_continuous,n_bin,target_col,filename):
    ''' This User Defined Function performs the following 
    1. Intiialize Excel Workbook
    2. Calls cont_bin_Miss() or cont_bin_NO_Miss() to perform binning based on the n_bin provided 
    3. Write to excel with the binning information'''
    
    global continuous_target_nx,df_fnl
    
    '''Initialize Excel Work Book for writing'''
    df_continuous_column = list(df_continuous.columns)
    writer = pd.ExcelWriter(filename,engine='xlsxwriter')
    workbook=writer.book
    worksheet=workbook.add_worksheet('Continous')
    writer.sheets['Continous'] = worksheet
    n = 1
    
    for i in range(len(df_continuous_column)):
        if (df_continuous_column[i] != target_col): 
            '''calls create_volume_group() to create bins - equal volume of bins with unique 
            values (n_bin - > it indicates the number of bins to be created) '''
            create_volume_group(df_continuous,df_continuous_column[i],target_col,n_bin)
            df_fnl=df_fnl[[df_continuous_column[i],target_col,'Decile']]
            
            ''' Checking the Continous Variable is having Missing Values '''
            if df_fnl.eval(df_continuous_column[i]).isnull().sum() > 0:
                '''calls cont_bin_Miss() for the conitnous variable having missing value'''
                cont_bin_Miss(df_fnl,df_continuous_column,i,target_col,filename)                
            else:
                '''calls cont_bin_NO_Miss() for the conitnous variable having NO missing value'''
                cont_bin_NO_Miss(df_fnl,df_continuous_column,i,target_col,filename)
            
            '''Write to excel with the binning information'''
            worksheet.write_string(n, 0, df_continuous_column[i])
            continuous_target_nx.to_excel(writer,sheet_name='Continous',startrow=n+1 , startcol=0,index = False)
            n += len(continuous_target_nx.index) + 10
           
    writer.save()
        

```
# Categorical Feature Rank & Plot


```python
from Categorical_Field_Rank_Plot_Regression import *
cat_bin(df,df_cat,target_col=target_nm,filename="Categorical_Base.xlsx")       
add_table_plot(df,in_file= "Categorical_Base.xlsx",sheet_nm='Categorical',target_col=target_nm,out_file="Categorical_Rank_Plot.xlsx")
```

# *****************************************
# Categorical_Field_Rank_Plot_Regression.py
# *****************************************
```python
 ````Categorical field Descriptive statistics with PLOT for Regression Model```
import xlsxwriter
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mimg
import numpy as np
import os
import math

''' Creating two new directory in the working directory to store the Plots of variables ''' 
work_dir=os.getcwd()
print(work_dir)
new_dir=work_dir+"/line_plot/"
check_dir_present=os.path.isdir(new_dir)
if check_dir_present == False:
    os.mkdir(new_dir)
    print("New Directory created : " + str(new_dir))
else:
    print("Existing Directory used : " + str(new_dir))
    
new_dir=work_dir+"/scatter_plot/"
check_dir_present=os.path.isdir(new_dir)
if check_dir_present == False:
    os.mkdir(new_dir)
    print("New Directory created : " + str(new_dir))
else:
    print("Existing Directory used : " + str(new_dir))


def plot_stat(df_categorical,title_name,target_col):
    '''This User Defined Functions performs following
    1. Line Plot showing Volume% and Average Call Volume % against 'Levels'
    2. Scatter Plot showing Target against 'Levels' of the Categorical Variables'''
    
    global df_plot,file_i,img_name,work_dir,scatter_img_name
    
    ''' Line Plot '''    
    width=0.35

    df_plot.plot(x='Levels',y='Volume(%)',kind='bar',width=width,label=('Volume(%)'),color='b')
    plt.ylabel('Volume(%)',color='b')
    plt.ylim((0,2))
    plt.legend(loc='upper right')

    df_plot['Avg_Call_Volume'].plot(secondary_y=True,label=('Avg_Call_Volume'),color='r',rot=90)
    plt.ylabel('Avg_Call_Volume',color='r')
    plt.ylim((0,2))
    plt.legend(loc='upper right')
    
    axis_1=plt.gca()

    for i,j in zip(df_plot['Volume(%)'].index,df_plot['Volume(%)']):
        i=round(i,2)
        j=round(j,2)
        axis_1.annotate('%s' %j,xy=(i,j),color='k')
        axis_1.annotate('%s' %i,xy=(i,j),color='k')
        
    for i,j in zip(df_plot['Avg_Call_Volume'].index,df_plot['Avg_Call_Volume']):
        i=round(i,2)
        j=round(j,2)
        axis_1.annotate('%s' %j,xy=(i,j),color='k')
        axis_1.annotate('%s' %i,xy=(i,j),color='k')

    plt.xlim([-width,len(df_plot['Volume(%)'])-width])
    plt.title(title_name)
    plt.xlabel('Levels')
    plt.grid()
    img_name=str(work_dir)+str("/line_plot/")+str(title_name)+ str(".png")
    plt.savefig(img_name,dpi=300,bbox_inches='tight')
    plt.clf()
    
    ''' Scatter Plot '''
    fig=plt.figure(figsize=(6,4))
    df_categorical[title_name]=df_categorical[title_name].fillna("Missing")
    plt.scatter(df_categorical[title_name],df_categorical[target_col],c='DarkBlue')
    plt.ylabel(target_col,color='b')
    plt.xlabel(title_name,color='b')
    plt.xticks(rotation=90)
    plt.title(title_name)
    plt.grid()
    scatter_img_name=str(work_dir)+str("/scatter_plot/")+str(title_name)+ str(".png")
    fig.savefig(scatter_img_name,dpi=300,bbox_inches='tight')
    plt.clf()

def add_table_plot(df_categorical,in_file,sheet_nm,out_file,target_col):
    ''' This User Defined Function adds Line Plot and Scatter Plot in an excel 
    just on the beside of the binning data of the categorical Fields.
    This gives better readability in analysing the data.
    '''
    
    global df_plot,img_name,work_dir,scatter_img_name
    
    work_dir=os.getcwd()
    df_cont=pd.read_excel(in_file,header=None,sheet_name=sheet_nm)
    df_cont.columns=['Levels','Avg_Call_Volume','TOTAL','Volume(%)']
    df_cont=df_cont.fillna('')
    wb=xlsxwriter.Workbook(out_file)
    ws=wb.add_worksheet(sheet_nm)
    
    for i in range(len(df_cont)):
        
        for j in range(len(df_cont.columns)):
            if j == 0:
                col_pos= 'A'
            if j == 1:
                col_pos= 'B'
            if j == 2:
                col_pos= 'C'  
            if j == 3:
                col_pos= 'D'
            if j == 4:
                col_pos= 'E'
            if j == 5:
                col_pos= 'F'
            if j == 6:
                col_pos= 'G'  
            if j == 7:
                col_pos= 'H'
            if j == 8:
                col_pos= 'I'

            wrt_pos=str(col_pos)+str(i)
            ws.write(wrt_pos,df_cont.iloc[i,j])
            
        if df_cont.iloc[:,0][i] == "Levels":
            pos_min_loc=i+int(1)
            if pos_min_loc==1:
                title_name=df_cont.columns[0]
            else:
                title_loc=int(pos_min_loc)-int(2)
                title_name=df_cont.iloc[:,0][title_loc]
        
        if df_cont.iloc[:,0][i] == "Total":
            pos_max_loc=i
            df_plot=df_cont[pos_min_loc:pos_max_loc]
            df_plot.columns=['Levels','Avg_Call_Volume','TOTAL','Volume(%)']
            df_plot=df_plot.reset_index()
            ''' Calls plot_stat() to create line plot and scatter plot '''
            plot_stat(df_categorical,title_name,target_col)
            img_pos=str('H')+ str(pos_min_loc-int(2))   

            ''' Inset line plot in the excel '''
            ws.insert_image(img_pos,img_name,{'x_scale' : 0.6, 'y_scale' : 0.6})

            ''' Insert Scatter plot in the excel '''
            scatter_img_pos=str('N')+ str(pos_min_loc-int(2))  
            ws.insert_image(scatter_img_pos,scatter_img_name,{'x_scale' : 0.6, 'y_scale' : 0.6})
    wb.close()

    
def cat_bin_trend(df_cat_fnl,df_categorical_column,i,target_col,filename):
    '''This User defined function creates bins on the basis of parameter n_bin (number of bins)
    provided'''

    global categorical_target_nx

    categorical_target_nx = pd.DataFrame(
        [df_cat_fnl.groupby('Levels')[target_col].mean(),
         df_cat_fnl.groupby('Levels')[df_categorical_column[i]].count()]
        ).T

    categorical_target_nx.columns = ["Avg_Call_Volume","TOTAL"]
    categorical_target_nx['Avg_Call_Volume'] = categorical_target_nx['Avg_Call_Volume'].fillna(0)
    categorical_target_nx['TOTAL'] = categorical_target_nx['TOTAL'].fillna(0)

    categorical_target_nx=categorical_target_nx.reset_index()
    categorical_target_nx = categorical_target_nx.rename(columns={categorical_target_nx.columns[0]: "Levels"})    
        
    list_vol_pct=[]

    for j in range(len(categorical_target_nx)):

        list_vol_pct.append(categorical_target_nx['TOTAL'][j]/categorical_target_nx['TOTAL'].sum())
    
    categorical_target_nx = pd.concat([categorical_target_nx,pd.Series(list_vol_pct)],axis=1)
    categorical_target_nx = categorical_target_nx[["Levels","Avg_Call_Volume","TOTAL",0]]
    categorical_target_nx = categorical_target_nx.rename(columns={categorical_target_nx.columns[len(categorical_target_nx.keys())-1]: "Volume(%)"})
    categorical_target_nx = categorical_target_nx.sort_values(by=['TOTAL'], ascending=False)

    categorical_target_nx = categorical_target_nx.append({"Levels":"Total",
                                     "Avg_Call_Volume":df_cat_fnl[target_col].mean(),
                                     "TOTAL":categorical_target_nx['TOTAL'].sum(),
                                     "Volume(%)":categorical_target_nx['Volume(%)'].sum()
                                                           },ignore_index=True)

def cat_bin(df,df_categorical,target_col,filename):
    ''' This User Defined Function performs the following 
    1. Intiialize Excel Workbook
    2. Calls cat_bin_trend to perform binning based on the n_bin provided 
    3. Write to excel with the binning information'''

    global categorical_target_nx

    df_categorical_column = list(df_categorical.columns)
   
    '''Initialize Excel Work Book for writing'''
    writer = pd.ExcelWriter(filename,engine='xlsxwriter')
    workbook=writer.book
    worksheet=workbook.add_worksheet('Categorical')
    writer.sheets['Categorical'] = worksheet
    n = 1    
    
    for i in range(len(df_categorical_column)):

        if (df_categorical_column[i] != target_col):       
            '''Replaces NAN Value with 'Missing' level'''
            nparray_cat=df[df_categorical_column[i]].fillna('Missing').unique()
            nparray_sort=np.sort(nparray_cat)        
            df_cat = pd.concat([pd.Series(nparray_sort),pd.Series(nparray_sort)],axis=1, keys=[df_categorical_column[i],'Levels'])        
            df_tst = df.loc[:, [df_categorical_column[i],target_col]].sort_values(by=[df_categorical_column[i]]).fillna('Missing')
            df_cat_fnl = pd.merge(df_tst, df_cat, how='left', on=[df_categorical_column[i]])                 
            '''Calls cat_bin_trend to perform binning based on the n_bin provided '''
            cat_bin_trend(df_cat_fnl,df_categorical_column,i,target_col,filename)                                   
            '''Write to excel with the binning information'''
            worksheet.write_string(n, 0, df_categorical_column[i])
            categorical_target_nx.to_excel(writer,sheet_name='Categorical',startrow=n+1 , startcol=0,index = False)
            n += len(categorical_target_nx.index) + 10

    writer.save()      
```

# Continuous Feature WOE,IV


```python
from Continuous_Field_WOE_IV_Regression import *
cont_bin(df,df_cont,n_bin=10,target_col=target_nm,filename="Continuous_WOE.xlsx")
```
# *****************************************
# Continuous_Field_WOE_IV_Regression.py
# *****************************************
```python
import xlsxwriter
import pandas as pd
import numpy as np
import os
import math

################ Final Version : Continuous Field  : Bin Based IV calculation 

def create_volume_group(df_continuous,curr_var,target_col,n_bin):
    '''This User defined function creates bins on the basis of parameter n_bin (number of bins) provided
    This algorithm creates almost eqi_volume groups with unique values in groups : 
    1. It calculates the Average Bin Volume by dividing the total volume of data by number of bins
    2. It sorts the data based on the value of the continous variables
    3. It directly moves to index (I1) having value Average Bin Volume (ABV) 
    4. It checks the value of continous variable at the index position decided in previosu step
    5. It finds the index(I2) of last position of the value identified at previous step 4
    6. It concludes the data of the First Bin within the range (0 to I2)
    7. The Index I1 is again calculated as I1 = I2 + ABV and step 4-6 is repeated
    8. This Process is continued till the desired number of bins are created
    9. Seperate Bin is created if the continuous variable is having missing value
    
    Note : qcut() does provide equi-volume groups but does not provide unique values in the groups.
    hence,qcut() is not used in binning the data.
    '''
    
    global df_fnl
    df_continuous_column=[]
    df_continuous_column.append(curr_var)
    ttl_vol=len(df_continuous)
    avg_bin_vol = round(ttl_vol/n_bin)
    lst=[]
    df_fnl=pd.DataFrame(lst)
    df_mod=pd.DataFrame(lst)
    df_mod_null=pd.DataFrame(lst)

    for i in range(len(df_continuous_column)):
        if (df_continuous_column[i] != target_col):
            curr_var=df_continuous_column[i]

            df_mod1=pd.DataFrame(df_continuous[[curr_var,target_col]])

            #### Sort the Data ####
            df_mod1=df_mod1.sort_values(by=[curr_var],ascending=True)
            df_mod1=df_mod1.reset_index()
        
            df_mod_null=df_mod1[pd.isnull(df_mod1[curr_var])]
            df_mod1=df_mod1.dropna(subset=[curr_var]) 
        
            seq=list(range(1,len(df_mod1)+int(1)))
            df_seq=pd.DataFrame(seq,columns=['sequence'])

            df_mod1=pd.concat([df_mod1,df_seq],axis=1)
        
            '''Creating the Missing BIN'''
            if len(df_mod_null) > int(0):
                ttl_vol=len(df_mod1)
                avg_bin_vol = round(ttl_vol/n_bin)
                group_num='missing'

        
            pos_indx_max = 0
            val_list=df_mod1[curr_var].unique()
            
            '''Checks if the Unique Values of the continuos variable is 2,then simply create 2 bins'''       
            if len(val_list) == 2:
                for i in range(len(val_list)):
                    val_of_indx=val_list[i]
                    df_mod3=df_mod1[df_mod1[curr_var] == val_of_indx]
                    group_num=i+int(1)
                    df_mod3['Decile']=group_num
                    df_fnl=pd.concat([df_fnl,df_mod3])

            '''Checks if the Unique Values of the continuos variable is more than 2'''
            if len(val_list) != 2:        
                for bin_num in range(n_bin):
                    if  pos_indx_max < ttl_vol-1 :
                        '''For the First Group, index is assigned to length of average bin volume'''
                        if bin_num == 0:
                            indx = (bin_num+int(1)) * avg_bin_vol
                            '''Next Groups:index = length of average bin volume plus the index of last group'''
                        else:
                            indx = int(pos_indx_max) + int(avg_bin_vol)
                
                        '''Setting the Index for last Group'''
                        if indx > ttl_vol:
                            indx = ttl_vol-int(1)

                        val_of_indx = df_mod1[curr_var].iat[indx]
                                        
                        if math.isnan(val_of_indx) == True:
                            pos_indx_min=pos_indx_max+int(1)
                            pos_indx_max=ttl_vol-int(1)
                        else:
                            df_mod3=df_mod1[df_mod1[curr_var] == val_of_indx]
                            pos_indx_min = pos_indx_max
                            pos_indx_max = df_mod3['sequence'].iat[-1]
                       
                        df_mod3=df_mod1[pos_indx_min:pos_indx_max]
                        group_num=bin_num+int(1)
                        df_mod3['Decile']=group_num
                    
                        df_fnl=pd.concat([df_fnl,df_mod3])
            
            df_fnl=pd.concat([df_fnl,df_mod_null])

def cont_bin_Miss(df_dcl_fnl,df_continuous_column,i,target_col,filename):
    '''This User Defined Function creates BINS for continous variables having some Missing values'''
    global continuous_target_nx

    '''The Missing Values have been grouped together as a seperate bin - "Missing" '''
    df_dcl_fnl[df_continuous_column[i]] = df_dcl_fnl[df_continuous_column[i]].astype(object)
    df_dcl_fnl[df_continuous_column[i]] = df_dcl_fnl[df_continuous_column[i]].fillna('Missing')
    df_dcl_fnl['Decile'] = df_dcl_fnl['Decile'].fillna('Missing')
    

    continuous_target_nx = pd.DataFrame(
        [df_dcl_fnl.groupby('Decile')[df_continuous_column[i]].min(),
         df_dcl_fnl.groupby('Decile')[df_continuous_column[i]].max(),
         df_dcl_fnl.groupby('Decile')[target_col].sum(),
         df_dcl_fnl.groupby('Decile')[df_continuous_column[i]].count()]
        ).T
    continuous_target_nx.columns = ["MIN","MAX","Total_Event","TOTAL"]
    continuous_target_nx.Total_Event=continuous_target_nx.Total_Event.fillna(0)
    
    continuous_target_nx=continuous_target_nx.reset_index()
    list1=[]
    list_vol_pct=[]
    list_event_pct=[]
    for i in range(len(continuous_target_nx.MIN)):
        list1.append(str(continuous_target_nx['MIN'][i])+'-'+str(continuous_target_nx['MAX'][i]))
        list_vol_pct.append(continuous_target_nx['TOTAL'][i]/continuous_target_nx['TOTAL'].sum())
        list_event_pct.append(continuous_target_nx['Total_Event'][i]/continuous_target_nx['Total_Event'].sum())

    continuous_target_nx = pd.concat([pd.Series(list1),continuous_target_nx,pd.Series(list_vol_pct),pd.Series(list_event_pct)],axis=1)
    continuous_target_nx=continuous_target_nx.reindex(columns=["Decile","MIN","MAX",0,"Total_Event","TOTAL",1,2])
    continuous_target_nx = continuous_target_nx[["Decile","MIN","MAX",0,"Total_Event","TOTAL",1,2]]

    continuous_target_nx = continuous_target_nx.rename(columns={continuous_target_nx.columns[len(continuous_target_nx.keys())-5]: "Range"})
    continuous_target_nx = continuous_target_nx.rename(columns={continuous_target_nx.columns[len(continuous_target_nx.keys())-2]: "Volume(%)"})
    continuous_target_nx = continuous_target_nx.rename(columns={continuous_target_nx.columns[len(continuous_target_nx.keys())-1]: "Total_Event(%)"})
    continuous_target_nx = continuous_target_nx.rename(columns={continuous_target_nx.columns[len(continuous_target_nx.keys())-8]: "BINS"})       
  
    continuous_target_nx = continuous_target_nx.append({"BINS":"Total",
                                 "MIN":" ",
                                 "MAX":" ",
                                 "Range":" ",
                                 "Total_Event":continuous_target_nx['Total_Event'].sum(),
                                 "TOTAL":continuous_target_nx['TOTAL'].sum(),
                                 "Volume(%)":continuous_target_nx['Volume(%)'].sum(),
                                 "Total_Event(%)":continuous_target_nx['Total_Event(%)'].sum()
                                 },ignore_index=True)

    
def cont_bin_NO_Miss(df_dcl_fnl,df_continuous_column,i,target_col,filename):
    '''This User defined function creates BINS for continuous variables having no missing values'''
    global continuous_target_nx

    continuous_target_nx = pd.DataFrame(
        [df_dcl_fnl.groupby('Decile')[df_continuous_column[i]].min(),
         df_dcl_fnl.groupby('Decile')[df_continuous_column[i]].max(),
         df_dcl_fnl.groupby('Decile')[target_col].sum(),
         df_dcl_fnl.groupby('Decile')[df_continuous_column[i]].count()]
        ).T
    continuous_target_nx.columns = ["MIN","MAX","Total_Event","TOTAL"]

    continuous_target_nx=continuous_target_nx.reset_index()
    list1=[]
    list_vol_pct=[]
    list_event_pct=[]
    
    for i in range(len(continuous_target_nx.MIN)):
        list1.append(str(continuous_target_nx['MIN'][i])+'-'+str(continuous_target_nx['MAX'][i]))
        list_vol_pct.append(continuous_target_nx['TOTAL'][i]/continuous_target_nx['TOTAL'].sum())
        list_event_pct.append(continuous_target_nx['Total_Event'][i]/continuous_target_nx['Total_Event'].sum())

    continuous_target_nx = pd.concat([pd.Series(list1),continuous_target_nx,pd.Series(list_vol_pct),pd.Series(list_event_pct)],axis=1)
    continuous_target_nx = continuous_target_nx[["Decile","MIN","MAX",0,"Total_Event","TOTAL",1,2]]
    continuous_target_nx = continuous_target_nx.rename(columns={continuous_target_nx.columns[len(continuous_target_nx.keys())-5]: "Range"})
    continuous_target_nx = continuous_target_nx.rename(columns={continuous_target_nx.columns[len(continuous_target_nx.keys())-2]: "Volume(%)"})
    continuous_target_nx = continuous_target_nx.rename(columns={continuous_target_nx.columns[len(continuous_target_nx.keys())-1]: "Total_Event(%)"})
    continuous_target_nx = continuous_target_nx.rename(columns={continuous_target_nx.columns[len(continuous_target_nx.keys())-8]: "BINS"})

    continuous_target_nx = continuous_target_nx.append({"BINS":"Total",
                                 "MIN":" ",
                                 "MAX":" ",
                                 "Range":" ",
                                 "Total_Event":continuous_target_nx['Total_Event'].sum(),
                                 "TOTAL":continuous_target_nx['TOTAL'].sum(),
                                 "Volume(%)":continuous_target_nx['Volume(%)'].sum(),
                                 "Total_Event(%)":continuous_target_nx['Total_Event(%)'].sum()
                                 },ignore_index=True)
    

def calc_iv(var_name):
    '''Calculates the Wieght of Evidence (WOE) and Information Value(IV) for Continuous fields'''
    global continuous_target_nx
    global data_fnl
    global IV_lst
    lst=[]
    
    continuous_target_nx.Total_Event=continuous_target_nx.Total_Event.fillna(0)
    row_cnt_without_total=len(continuous_target_nx)-int(1)
    
    for i in range(len(continuous_target_nx)):
        data_bin = continuous_target_nx.BINS[i]
        data_min = continuous_target_nx.MIN[i]
        data_max = continuous_target_nx.MAX[i]
        data_Value = continuous_target_nx.Range[i]
        data_All = int(continuous_target_nx.TOTAL[i])
        data_Target = int(continuous_target_nx.Total_Event[i])
        data_All_Pct = continuous_target_nx['Volume(%)'].iloc[i]
        data_Target_Pct = continuous_target_nx['Total_Event(%)'].iloc[i]

        data_WoE = np.log(data_Target_Pct / data_All_Pct)
        
        if (data_WoE == np.inf) or (data_WoE == -np.inf):
            data_WoE = 0
            
        data_IV = data_WoE * (data_Target_Pct - data_All_Pct)
        data=[data_bin,data_min,data_max,data_Value,data_Target,data_All,data_All_Pct,data_Target_Pct,data_WoE,data_IV]

        lst.append(data)
    
    data_fnl = pd.DataFrame(lst,columns=['Bins', 'Min', 'Max', 'Range', 'Total_Event', 'TOTAL', 'Volume(%)','Total_Event(%)','WOE','IV'])
    iv_val=[var_name,data_fnl['IV'].head(row_cnt_without_total).sum()]
    IV_lst.append(iv_val)
    
def cont_bin(df,df_continuous,n_bin,target_col,filename):
    ''' This User Defined Function performs the following 
    1. Intiialize Excel Workbook
    2. Calls cont_bin_Miss() or cont_bin_NO_Miss() to perform binning based on the n_bin provided 
    3. Write to excel with the binning information'''
    
    global continuous_target_nx
    global data_fnl
    global IV_lst
    global df_fnl
    
    df_continuous_column = list(df_continuous.columns)
    
    '''Initialize Excel Work Book for writing'''
    IV_lst=[]
    writer1 = pd.ExcelWriter(filename,engine='xlsxwriter')
    workbook=writer1.book
    worksheet=workbook.add_worksheet('WOE')
    writer1.sheets['WOE'] = worksheet
    n = 0
    m = -1
    
    for i in range(len(df_continuous_column)):
        if (df_continuous_column[i] != target_col):
            '''calls create_volume_group() to create bins - equal volume of bins with unique
            values <n_bin - > it indicates the number of bins to be created> '''
            create_volume_group(df_continuous,df_continuous_column[i],target_col,n_bin)
            df_fnl=df_fnl[[df_continuous_column[i],target_col,'Decile']]

            ''' Checking the Continous Variable is having Missing Values '''
            if df_fnl.eval(df_continuous_column[i]).isnull().sum() > 0:
                '''calls cont_bin_Miss() for the conitnous variable having missing value'''       
                cont_bin_Miss(df_fnl,df_continuous_column,i,target_col,filename)                
            else:
                '''calls cont_bin_NO_Miss() for the conitnous variable having NO missing value'''
                cont_bin_NO_Miss(df_fnl,df_continuous_column,i,target_col,filename)
            
            calc_iv(df_continuous_column[i])

            worksheet.write_string(n, 0, df_continuous_column[i])

            '''Write to excel with WOE in worksheet "WOE" '''
            data_fnl.to_excel(writer1,sheet_name='WOE',startrow=n+1 , startcol=0,index = False)
            n += len(continuous_target_nx.index) + 4
    
    data_IV = pd.DataFrame(IV_lst,columns=['Variable','IV_value'])
    '''Write the IV to excel worksheet "IV" '''
    data_IV.to_excel(writer1,sheet_name='IV',startrow=m+1 , startcol=0,index = False)
    print(data_IV)
    writer1.save()
        
       
    
```

# Categorical Feature WOE,IV


```python
from Categorical_Field_WOE_IV_Regression import * 
cat_bin(df,df_cat,target_col=target_nm,filename="Categorical_WOE.xlsx")
df=automate_woe_population(df,df_cat,filename="Categorical_WOE.xlsx")
```

# *****************************************
# Categorical_Field_WOE_IV_Regression.py
# *****************************************
```python
 ```Categorical Field : Weight of Evidence and IV calcualtion for Regression Model '''
import xlsxwriter
import os
import math
import pandas as pd
import numpy as np

def calc_iv(var_name):
    '''Calculates the Wieght of Evidence (WOE) and Information Value(IV) for Catgeorical
    fields'''
    global categorical_target_nx
    global data_fnl
    global IV_lst
    lst=[]
    
    categorical_target_nx.Total_Event=categorical_target_nx.Total_Event.fillna(0)
    row_cnt_without_total=len(categorical_target_nx)-int(1)
    
    for i in range(len(categorical_target_nx)):
        data_bin = categorical_target_nx.Levels[i]
        data_Target = int(categorical_target_nx.Total_Event[i])
        data_All = int(categorical_target_nx.TOTAL[i])
        data_All_Pct = categorical_target_nx['Volume(%)'].iloc[i]
        data_Target_Pct = categorical_target_nx['Total_Event(%)'].iloc[i]

        data_WoE = np.log(data_Target_Pct / data_All_Pct)
        
        if (data_WoE == np.inf) or (data_WoE == -np.inf):
            data_WoE = 0
            
        data_IV = data_WoE * (data_Target_Pct - data_All_Pct)
        data=[data_bin,data_Target,data_All,data_All_Pct,data_Target_Pct,data_WoE,data_IV]

        lst.append(data)
    
    data_fnl = pd.DataFrame(lst,columns=['Levels', 'Total_Event', 'TOTAL', 'Volume(%)', 'Total_Event(%)', 'WOE','IV'])
    iv_val=[var_name,data_fnl['IV'].head(row_cnt_without_total).sum()]
    IV_lst.append(iv_val)

    
def cat_bin_trend(df_cat_fnl,df_categorical_column,i,target_col,filename):
    '''This User defined function creates the bins / groups on the 'Levels' of the 
    Categorical Columns
    1. Event -> Target = 1
    2. Non-Event -> Target = 0 
    3. ALong with the Levels of the categorical Columns, A summary record is also created with
    header "TOTAL" '''

    global categorical_target_nx

    categorical_target_nx = pd.DataFrame(
        [df_cat_fnl.groupby('Levels')[target_col].sum(),
         df_cat_fnl.groupby('Levels')[df_categorical_column[i]].count()]
        ).T

    categorical_target_nx.columns = ["Total_Event","TOTAL"]

    categorical_target_nx=categorical_target_nx.reset_index()
    categorical_target_nx = categorical_target_nx.rename(columns={categorical_target_nx.columns[0]: "Levels"})    
        
    list_vol_pct=[]
    list_event_pct=[]

    for j in range(len(categorical_target_nx.Total_Event)):

        list_vol_pct.append(categorical_target_nx['TOTAL'][j]/categorical_target_nx['TOTAL'].sum())
        list_event_pct.append(categorical_target_nx['Total_Event'][j]/categorical_target_nx['Total_Event'].sum())
    
    categorical_target_nx = pd.concat([categorical_target_nx,pd.Series(list_vol_pct),pd.Series(list_event_pct)],axis=1)
    
    
    categorical_target_nx = categorical_target_nx[["Levels","Total_Event","TOTAL",0,1]] 
    categorical_target_nx = categorical_target_nx.rename(columns={categorical_target_nx.columns[len(categorical_target_nx.keys())-2]: "Volume(%)"})
    categorical_target_nx = categorical_target_nx.rename(columns={categorical_target_nx.columns[len(categorical_target_nx.keys())-1]: "Total_Event(%)"})
    categorical_target_nx = categorical_target_nx.sort_values(by=['TOTAL'], ascending=False)
    
    categorical_target_nx = categorical_target_nx.append(
        {"Levels":"Total",
         "Total_Event":categorical_target_nx['Total_Event'].sum(),
         "TOTAL":categorical_target_nx['TOTAL'].sum(),
         "Volume(%)":categorical_target_nx['Volume(%)'].sum(),
         "Total_Event(%)":categorical_target_nx['Total_Event(%)'].sum()
        },ignore_index=True)
                 

def cat_bin(df,df_categorical,target_col,filename):
    ''' This User Defined Function performs the following :
    1. Replace the NAN value with "Missing" value 
    2. Binning is done on the unique Labels of the Categorical columns
    3. For Missing Values, it has been treated as seperate Label - "Missing"
    4. Calculates Weight of Evidence (WOE) of each bin/label of Categorical Variables
    5. Calculates Information Vale (IV) for each Categorical Variables  '''

    global categorical_target_nx,data_fnl,IV_lst
    
    df_categorical_column = list(df_categorical.columns)
    
    '''Initialization of list and excel workbook'''
    IV_lst=[]
    writer1 = pd.ExcelWriter(filename,engine='xlsxwriter')
    workbook=writer1.book
    worksheet=workbook.add_worksheet('WOE')
    writer1.sheets['WOE'] = worksheet
    n = 0
    m = -1  
    
    for i in range(len(df_categorical_column)):

        if (df_categorical_column[i] != target_col):
            '''Repplacing the NAN Value with "Missing" Value for treating the Missing Value as
            seperate bin/group'''
            nparray_cat=df[df_categorical_column[i]].fillna('Missing_need_replace').unique()
            nparray_sort=np.sort(nparray_cat)        
            df_cat = pd.concat([pd.Series(nparray_sort),pd.Series(nparray_sort)],axis=1, keys=[df_categorical_column[i],'Levels'])        
            df_tst = df.loc[:, [df_categorical_column[i],target_col]].sort_values(by=[df_categorical_column[i]]).fillna('Missing_need_replace')
            df_cat_fnl = pd.merge(df_tst, df_cat, how='left', on=[df_categorical_column[i]])                 
            ''' Creates Groups for each of the unique values of categorical variables '''
            cat_bin_trend(df_cat_fnl,df_categorical_column,i,target_col,filename)                             
            ''' Calculates WOE and IV '''
            calc_iv(df_categorical_column[i])
     
            ''' Writing the WOE in seperate worksheet "WOE" of Final Excel '''
            worksheet.write_string(n, 0, df_categorical_column[i])
            data_fnl.to_excel(writer1,sheet_name='WOE',startrow=n+1 , startcol=0,index = False)
            n += len(categorical_target_nx.index) + 4
    
    ''' Writing the IV in seperate worksheet "IV" of Final Excel '''
    data_IV = pd.DataFrame(IV_lst,columns=['Variable','IV_value'])
    data_IV.to_excel(writer1,sheet_name='IV',startrow=m+1 , startcol=0,index = False)       
    writer1.save()

def automate_woe_population(df,df_categorical,filename):
    '''This User Defined Function creates a new field with suffix "_WOE" and gets populated with
    Weight of Evidence as obtained for each 'Levels' of the Categorical Variables'''
    
    ''' Input : 
    1. df -> The Total Dataframe
    2. df_categorical -> The Dataframe having only Categorical columns
    3. filename -> The Excel File having WOE details for Categorical column 
    (created in previous step)
    '''
    
    ''' Return
    1. Updated DataFrame with new columnd with suffix "_WOE" having WOE values
    '''
    
    import pandas as pd
    woe_df=pd.read_excel(filename,sheet_name='WOE',header=None)

    for cat_i in list(df_categorical.columns):
        match_fnd=""
        new_col=str(cat_i)+ "_WOE"

        for j in range(len(woe_df)):        
            if str(cat_i) == str(woe_df.iloc[j][0]) and match_fnd =="":            
                match_fnd='y'

            if str(woe_df.iloc[j][0]) == "Total":
                match_fnd = ""

            if match_fnd == 'y':            
                if (str(woe_df.iloc[j][0]) != "Levels") :
                    #print(woe_df.iloc[j])
                    if (str(cat_i) == str(woe_df.iloc[j][0])):
                        woe_ln=str("df['") + str(cat_i) + str("_WOE']=0")
                    else:

                        if  str(woe_df.iloc[j][0]) == "Missing_need_replace":
                            woe_ln = str("df.loc[df['") + str(cat_i) + str("'].isna()")  + str(",'") + str(cat_i) + str("_WOE']=") + str(woe_df.iloc[j][5])

                            df.loc[df[cat_i].isna(),new_col]=woe_df.iloc[j][5]

                        else:
                            woe_ln = str("df.loc[df['") + str(cat_i) + str("']==") + str('"') + str(woe_df.iloc[j][0]) + str('"') + str(",'") + str(cat_i) + str("_WOE']=") + str(woe_df.iloc[j][5])

                            df.loc[df[cat_i]==str(woe_df.iloc[j][0]),new_col]=woe_df.iloc[j][5]



                    print(woe_ln)

    return df
```
