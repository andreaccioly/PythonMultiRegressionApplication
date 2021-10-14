'''
This script perfoms the basic process for applying a machine learning
algorithm to a dataset using Python libraries.

The four steps are:
   1. Download a dataset (using pandas)
   2. Process the numeric data (using numpy)
   3. Train and evaluate learners (using scikit-learn)
   4. Plot and compare results (using matplotlib)


The data is downloaded from URL, which is defined below. As is normal
for machine learning problems, the nature of the source data affects
the entire solution. When you change URL to refer to your own data, you
will need to review the data processing steps to ensure they remain
correct.

============
Example Data
============
The example is from https://web.archive.org/web/20180322001455/http://mldata.org/repository/data/viewslug/stockvalues/
It contains stock prices and the values of three indices for each day
over a five year period. See the linked page for more details about
this data set.

This script uses regression learners to predict the stock price for
the second half of this period based on the values of the indices. This
is a naive approach, and a more robust method would use each prediction
as an input for the next, and would predict relative rather than
absolute values.
'''

# Remember to update the script for the new data when you change this URL
#URL = "https://raw.githubusercontent.com/microsoft/python-sklearn-regression-cookiecutter/master/stockvalues.csv"
URL = "H:\\Users\\admin\\Projects\\Desafios\\DesafioIBM\\Arquivos\\registros-prod.xlsx"

# This is the column of the sample data to predict.
# Try changing it to other integers between 1 and 155.
TARGET_COLUMN = 5#32

# Uncomment this call when using matplotlib to generate images
# rather than displaying interactive UI.
import matplotlib
# matplotlib.use('Agg')

from pandas import read_table
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_excel
from feature_engine.encoding import CountFrequencyEncoder
from sklearn import linear_model
import statsmodels.api as sm
import tkinter as tk 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg



try:
    # [OPTIONAL] Seaborn makes plots nicer
    import seaborn
except ImportError:
    pass

# =====================================================================

def download_data():
    '''
    Downloads the data for this script into a pandas DataFrame.
    '''

    # If your data is in an Excel file, install 'xlrd' and use
    # pandas.read_excel instead of read_table
    
    frame = read_excel(URL, engine='openpyxl')

    # If your data is in a private Azure blob, install 'azure-storage' and use
    # BlockBlobService.get_blob_to_path() with read_table() or read_excel()
    #from azure.storage.blob import BlockBlobService
    #service = BlockBlobService(ACCOUNT_NAME, ACCOUNT_KEY)
    #service.get_blob_to_path(container_name, blob_name, 'my_data.csv')
    #frame = read_table('my_data.csv', ...

        #frame = read_table(URL)
        
        # Uncomment if the file needs to be decompressed
        #compression='gzip',
        #compression='bz2',

        # Specify the file encoding
        # Latin-1 is common for data from US sources
        #encoding='latin-1',
        #encoding='utf-8',  # UTF-8 is also common

        # Specify the separator in the data
        #sep=',',            # comma separated values
        #sep='\t',          # tab separated values
        #sep=' ',           # space separated values

        # Ignore spaces after the separator
        #skipinitialspace=True,

        # Generate row labels from each row number
        #index_col=None,
        #index_col=0,       # use the first column as row labels
        #index_col=-1,      # use the last column as row labels

        # Generate column headers row from each column number
        #header=None,
        #header=0,          # use the first line as headers

        # Use manual headers and skip the first row in the file
        #header=0,
        #names=['col1', 'col2', ...],
    #)

    # Return the entire frame
    return frame

    # Return a subset of the columns
    #return frame[[156, 157, 158, TARGET_COLUMN]]


# =====================================================================


def get_features_and_labels(frame):
    '''
    Transforms and scales the input data and returns numpy arrays for
    training and testing inputs and targets.
    '''

    # Replace missing values with 0.0
    # or we can use scikit-learn to calculate missing values below
    #frame[frame.isnull()] = 0.0

    #Preprocess Categorycal Variables
    encoder = CountFrequencyEncoder(encoding_method='frequency')

    # fit the encoder
    encoder.fit(frame)

    # Transform Data
    global newframe 
    newframe = encoder.transform(frame)        

    # Convert values to floats
    arr = np.array(newframe, dtype=np.float)

    # Normalize the entire data set
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    arr = MinMaxScaler().fit_transform(arr)

    # Use the last column as the target value
    X, y = arr[:, :-1], arr[:, -1]
    # To use the first column instead, change the index value
    #X, y = arr[:, 1:], arr[:, 0]
    
    # Use 50% of the data for training, but we will test against the
    # entire set
    from sklearn.model_selection import train_test_split
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3)
    X_test, y_test = X, y
    
    # If values are missing we could impute them from the training data
    #from sklearn.preprocessing import Imputer
    #imputer = Imputer(strategy='mean')
    #imputer.fit(X_train)
    #X_train = imputer.transform(X_train)
    #X_test = imputer.transform(X_test)
    
    # Normalize the attribute values to mean=0 and variance=1
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    # To scale to a specified range, use MinMaxScaler
    #from sklearn.preprocessing import MinMaxScaler
    #scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Fit the scaler based on the training data, then apply the same
    # scaling to both training and test sets.
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Return the training and test sets
    return X_train, X_test, y_train, y_test

# =====================================================================


def evaluate_learner(X_train, X_test, y_train, y_test):
    '''
    Run multiple times with different algorithms to get an idea of the
    relative performance of each configuration.

    Returns a sequence of tuples containing:
        (title, expected values, actual values)
    for each learner.
    '''

    # Use a support vector machine for regression
    from sklearn.svm import SVR

    # Train using a radial basis function
    svr = SVR(kernel='rbf', gamma=0.1)
    svr.fit(X_train, y_train)
    y_pred = svr.predict(X_test)
    r_2 = svr.score(X_test, y_test)
    yield 'RBF Model ($R^2={:.3f}$)'.format(r_2), y_test, y_pred

    # Train using a linear kernel
    svr = SVR(kernel='linear')
    svr.fit(X_train, y_train)
    y_pred = svr.predict(X_test)
    r_2 = svr.score(X_test, y_test)
    yield 'Linear Model ($R^2={:.3f}$)'.format(r_2), y_test, y_pred

    # Train using a polynomial kernel
    svr = SVR(kernel='poly', degree=2)
    svr.fit(X_train, y_train)
    y_pred = svr.predict(X_test)
    r_2 = svr.score(X_test, y_test)
    yield 'Polynomial Model ($R^2={:.3f}$)'.format(r_2), y_test, y_pred


# =====================================================================


def plot(results):
    '''
    Create a plot comparing multiple learners.

    `results` is a list of tuples containing:
        (title, expected values, actual values)
    
    All the elements in results will be plotted.
    '''

    # Using subplots to display the results on the same X axis
    fig, plts = plt.subplots(nrows=len(results), figsize=(8, 8))
    fig.canvas.set_window_title('Predicting data from ' + URL)

    # Show each element in the plots returned from plt.subplots()
    for subplot, (title, y, y_pred) in zip(plts, results):
        # Configure each subplot to have no tick marks
        # (these are meaningless for the sample dataset)
        subplot.set_xticklabels(())
        subplot.set_yticklabels(())

        # Label the vertical axis
        subplot.set_ylabel('PESO_BOMBOM')

        # Set the title for the subplot
        subplot.set_title(title)

        # Plot the actual data and the prediction
        subplot.plot(y, 'b', label='actual')
        subplot.plot(y_pred, 'r', label='predicted')
        
        # Shade the area between the predicted and the actual values
        subplot.fill_between(
            # Generate X values [0, 1, 2, ..., len(y)-2, len(y)-1]
            np.arange(0, len(y), 1),
            y,
            y_pred,
            color='r',
            alpha=0.2
        )

        # Mark the extent of the training data
        subplot.axvline(len(y) // 2, linestyle='--', color='0', alpha=0.2)

        # Include a legend in each subplot
        subplot.legend()

    # Let matplotlib handle the subplot layout
    fig.tight_layout()

    # ==================================
    # Display the plot in interactive UI
    plt.show()

    # To save the plot to an image file, use savefig()
    #plt.savefig('plot.png')

    # Open the image file with the default image viewer
    #import subprocess
    #subprocess.Popen('plot.png', shell=True)

    # To save the plot to an image in memory, use BytesIO and savefig()
    # This can then be written to any stream-like object, such as a
    # file or HTTP response.
    #from io import BytesIO
    #img_stream = BytesIO()
    #plt.savefig(img_stream, fmt='png')
    #img_bytes = img_stream.getvalue()
    #print('Image is {} bytes - {!r}'.format(len(img_bytes), img_bytes[:8] + b'...'))

    # Closing the figure allows matplotlib to release the memory used.
    plt.close()


# =====================================================================
def gui(df, X_train, X_test, y_train, y_test):
    
    # with sklearn
    regr = linear_model.LinearRegression()
    regr.fit(X_test, y_test)

    # tkinter GUI
    root= tk.Tk()

    canvas1 = tk.Canvas(root, width = 500, height = 300)
    canvas1.pack()

    # with sklearn
    Intercept_result = ('Intercept: ', regr.intercept_)
    label_Intercept = tk.Label(root, text=Intercept_result, justify = 'center')
    canvas1.create_window(260, 220, window=label_Intercept)

    # with sklearn
    Coefficients_result  = ('Coefficients: ', regr.coef_)
    label_Coefficients = tk.Label(root, text=Coefficients_result, justify = 'center')
    canvas1.create_window(260, 240, window=label_Coefficients)

    # New_PESO_CHOC label and input box
    label1 = tk.Label(root, text='QTD_CHOC:')
    canvas1.create_window(100, 100, window=label1)
    entry1 = tk.Entry (root) # create 1st entry box
    canvas1.create_window(270, 100, window=entry1)

    # New_VAR_1 label and input box
    label2 = tk.Label(root, text=' VAR_1: ')
    canvas1.create_window(100, 120, window=label2)
    entry2 = tk.Entry (root) # create 2nd entry box
    canvas1.create_window(270, 120, window=entry2)

    # New_VAR_2 label and input box
    label3 = tk.Label(root, text='VAR_2: ')
    canvas1.create_window(100, 140, window=label3)
    entry3 = tk.Entry (root) # create 3rd entry box
    canvas1.create_window(270, 140, window=entry3)

    def values(): 
        global New_PESO_CHOC #our 1st input variable
        New_PESO_CHOC = float(entry1.get()) 
    
        global New_VAR_1 #our 2nd input variable
        New_VAR_1 = float(entry2.get()) 

        global New_VAR_2 # 3rd input variable 
        New_VAR_2 = float(entry3.get()) 
    
        Prediction_result  = ('Predicted PESO_BOMBOM: ', regr.predict([[New_Interest_Rate ,New_Unemployment_Rate]]))
        label_Prediction = tk.Label(root, text= Prediction_result, bg='orange')
        canvas1.create_window(260, 280, window=label_Prediction)
    
    button1 = tk.Button (root, text='Predict PESO_BOMBOM',command=values, bg='orange') # button to call the 'values' command above 
    canvas1.create_window(270, 170, window=button1)
 
    #plot 1st scatter 
    figure3 = plt.Figure(figsize=(5,4), dpi=100)
    ax3 = figure3.add_subplot(111)
    ax3.scatter(df['PESO_BOMBOM'].astype(float),df['QTD_CHOC'].astype(float), color = 'r')
    scatter3 = FigureCanvasTkAgg(figure3, root) 
    scatter3.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)
    ax3.legend(['QTD_CHOC']) 
    ax3.set_xlabel('PESO_BOMBOM')
    ax3.set_title('QTD_CHOC x PESO_BOMBOM')

    #plot 2nd scatter 
    figure4 = plt.Figure(figsize=(5,4), dpi=100)
    ax4 = figure4.add_subplot(111)
    ax4.scatter(df['PESO_BOMBOM'].astype(float),df['VAR_1'].astype(float), color = 'g')
    scatter4 = FigureCanvasTkAgg(figure4, root) 
    scatter4.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)
    ax4.legend(['VAR_1']) 
    ax4.set_xlabel('PESO_BOMBOM')
    ax4.set_title('VAR_1 x PESO_BOMBOM')

    #plot 3nd scatter 
    figure5 = plt.Figure(figsize=(5,4), dpi=100)
    ax4 = figure5.add_subplot(111)
    ax4.scatter(df['PESO_BOMBOM'].astype(float), df['VAR_2'].astype(float), color = 'b')
    scatter5 = FigureCanvasTkAgg(figure5, root) 
    scatter5.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)
    ax4.legend(['VAR_2']) 
    ax4.set_xlabel('PESO_BOMBOM')
    ax4.set_title('VAR_2 x PESO_BOMBOM')

    root.mainloop()

# ====================================================================================

if __name__ == '__main__':
    # Download the data set from URL
    print("Downloading data from {}".format(URL))
    frame = download_data()

    # Process data into feature and label arrays
    print("Processing {} samples with {} attributes".format(len(frame.index), len(frame.columns)))
    X_train, X_test, y_train, y_test = get_features_and_labels(frame)

    # Evaluate multiple regression learners on the data
    print("Evaluating regression learners")
    results = list(evaluate_learner(X_train, X_test, y_train, y_test))

    # Display the results
    print("Plotting the results")
    plot(results)

    #GUI for prediction
    print("Opening GUI")
    gui(newframe, X_train, X_test, y_train, y_test)
