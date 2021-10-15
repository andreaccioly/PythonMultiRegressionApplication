import numpy as np
import pandas as pd
import scipy.optimize as optimize
from feature_engine.encoding import CountFrequencyEncoder

#Home
#URL = "H:\\Users\\admin\\Projects\\Desafios\\DesafioIBM\\Arquivos\\analise-prescritiva.xlsx"
#Office
URL = "C:\\Users\\Andre.Vieira\\Downloads\\Teste\\analise-prescritiva.xlsx"

def download_data(URL):
    '''
    Downloads the data for this script into a pandas DataFrame.
    '''   
    df = pd.read_excel(URL)#, engine='openpyxl')
    print(df)
    # Return the entire frame
    return df

def qtd_choc(p,VAR_1,VAR_2):
    
    q = (p + (VAR_1 * 0.735) + (VAR_2 * -15.8443))/0.0123
    
    return q

def fun(p):

    fun = (200/(1 + np.exp(10 * (p - 9.5)))) + (200/(1 + np.exp(-0.8 * (p - 12))))

    return fun

def gradient_descent(x,y):
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
        print ("m {}, b {}, cost {} iteration {}".format(m_curr,b_curr,cost, i))
    
    return cost

df = download_data(URL)

#Preprocess Categorycal Variables
encoder = CountFrequencyEncoder(encoding_method='frequency')

# fit the encoder
encoder.fit(df)

# Transform Data
ndf = encoder.transform(df)

print(ndf)

# Convert values to floats
arr = np.array(ndf, dtype=np.float)

result_p = []

p = 10
for linha in arr:
    result_p.append(qtd_choc(p,linha[1],linha[2]))

print(result_p)

#x = np.array([1,2,3,4,5])
#y = np.array([5,7,9,11,13])

#gradient_descent(x,y)

    #def f(params):
    #    # print(params)  # <-- you'll see that params is a NumPy array
    #    a, b = params # <-- for readability you may wish to assign names to the component variables
    #    return a**2 + b**2 

    #initial_guess = [1, 1]
    #result = optimize.minimize(f, initial_guess)
    #if result.success:
    #    fitted_params = result.x
    #    print(fitted_params)
    #else:
    #    raise ValueError(result.message)