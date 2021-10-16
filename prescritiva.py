import numpy as np
import pandas as pd
from feature_engine.encoding import CountFrequencyEncoder
from sklearn import linear_model
from sklearn.model_selection import train_test_split 
from pandas import read_excel
import xlsxwriter


#Home
prescri = "H:\\Users\\admin\\Projects\\Desafios\\DesafioIBM\\Arquivos\\analise-prescritiva-qtd.xlsx"
regprod = "H:\\Users\\admin\\Projects\\Desafios\\DesafioIBM\\Arquivos\\registros-prod-work.xlsx"
resultado = "H:\\Users\\admin\\Projects\\Desafios\\DesafioIBM\\Arquivos\\resultado-analise-prescritiva.xlsx"
#Office
#URL = "C:\\Users\\Andre.Vieira\\Downloads\\Teste\\analise-prescritiva-teste.xlsx"

def download_data(URL):
    '''
    Downloads the data for this script into a pandas DataFrame.
    '''   
    df = read_excel(URL, engine='openpyxl')
    print(df)
    # Return the entire frame
    return df

def qtd_choc(p,VAR_1,VAR_2):
    
    q = (p + (VAR_1 * 0.735) + (VAR_2 * -15.8443))/0.0123
    
    return q

def multi_regression(df):

    X = df[['VAR_1' , 'VAR_21', 'PESO_BOMBOM' ]]
    y = df['QTD_CHOC']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
   
    model_ols = linear_model.LinearRegression(copy_X=True, fit_intercept=True, normalize=False)
    model_ols.fit(X_train,y_train)
     
    coef = model_ols.coef_
    intercept = model_ols.intercept_
    print('coef= ', coef)
    print('intercept= ', intercept)    

    return model_ols
    

def fun(p):

    return (200/(1 + np.exp(10 * (p - 9.5)))) + (200/(1 + np.exp(-0.8 * (p - 12))))
    

def gradient_descent(df):

    x = df['QTD_CHOC']  
    y = df['PESO_BOMBOM']
        
    m_curr = b_curr = 0
    iterations = 10000
    n = len(x)
    learning_rate = 0.08

    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = (1/n) * sum([val**2 for val in (y-y_predicted)])
        md = -(2/n)*sum(x*(y-y_predicted))
        bd = -(2/n)*sum(y-y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd        
    
    return print ("m {}, b {}, cost {} iteration {}".format(m_curr,b_curr,cost, i))

# Le dados
df = download_data(regprod) 
values = download_data(prescri)

# Regressao Linear
regr = multi_regression(df)

result = []

for index,row in values.iterrows():
        # Precição dos valores de QTD_CHOC para que o custo seja mínimo
        result.append(regr.predict([[row['VAR_1'], row['VAR_21'],10]]))

dfr = pd.DataFrame(result)
writer = pd.ExcelWriter(resultado, engine='xlsxwriter')
dfr.to_excel(writer)
writer.save()








