.. code:: ipython3

    import pandas as pd
    import os 
    

.. code:: ipython3

    dff=pd.read_csv("Downloads/samuel/fd.csv")
    dff




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Unnamed: 0.3</th>
          <th>Unnamed: 0.2</th>
          <th>Unnamed: 0.1</th>
          <th>Unnamed: 0</th>
          <th>Date</th>
          <th>Timestamp</th>
          <th>DeviceOrientation</th>
          <th>AccelerationX</th>
          <th>AccelerationY</th>
          <th>AccelerationZ</th>
          <th>Label</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0</td>
          <td>0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>2020-07-16 13:54:48</td>
          <td>1.594933e+09</td>
          <td>faceUp</td>
          <td>0.148743</td>
          <td>-0.035278</td>
          <td>-0.998413</td>
          <td>1</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1</td>
          <td>1</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>2020-07-16 13:54:48</td>
          <td>1.594933e+09</td>
          <td>faceUp</td>
          <td>0.102600</td>
          <td>0.023712</td>
          <td>-1.024200</td>
          <td>1</td>
        </tr>
        <tr>
          <th>2</th>
          <td>2</td>
          <td>2</td>
          <td>2.0</td>
          <td>2.0</td>
          <td>2020-07-16 13:54:48</td>
          <td>1.594933e+09</td>
          <td>faceUp</td>
          <td>0.076065</td>
          <td>-0.007690</td>
          <td>-0.985779</td>
          <td>1</td>
        </tr>
        <tr>
          <th>3</th>
          <td>3</td>
          <td>3</td>
          <td>3.0</td>
          <td>3.0</td>
          <td>2020-07-16 13:54:48</td>
          <td>1.594933e+09</td>
          <td>faceUp</td>
          <td>0.097046</td>
          <td>-0.025513</td>
          <td>-0.982819</td>
          <td>1</td>
        </tr>
        <tr>
          <th>4</th>
          <td>4</td>
          <td>4</td>
          <td>4.0</td>
          <td>4.0</td>
          <td>2020-07-16 13:54:48</td>
          <td>1.594933e+09</td>
          <td>faceUp</td>
          <td>0.100662</td>
          <td>-0.002991</td>
          <td>-1.000076</td>
          <td>1</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>164604</th>
          <td>164604</td>
          <td>108929</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>14-07-2020 15:26</td>
          <td>1.594766e+09</td>
          <td>faceUp</td>
          <td>-0.158188</td>
          <td>-0.138901</td>
          <td>-0.973404</td>
          <td>0</td>
        </tr>
        <tr>
          <th>164605</th>
          <td>164605</td>
          <td>108930</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>14-07-2020 15:26</td>
          <td>1.594766e+09</td>
          <td>faceUp</td>
          <td>-0.159973</td>
          <td>-0.145752</td>
          <td>-0.972702</td>
          <td>0</td>
        </tr>
        <tr>
          <th>164606</th>
          <td>164606</td>
          <td>108931</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>14-07-2020 15:26</td>
          <td>1.594766e+09</td>
          <td>faceUp</td>
          <td>-0.150299</td>
          <td>-0.138916</td>
          <td>-0.954269</td>
          <td>0</td>
        </tr>
        <tr>
          <th>164607</th>
          <td>164607</td>
          <td>108932</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>14-07-2020 15:26</td>
          <td>1.594766e+09</td>
          <td>faceUp</td>
          <td>-0.171616</td>
          <td>-0.141953</td>
          <td>-0.969955</td>
          <td>0</td>
        </tr>
        <tr>
          <th>164608</th>
          <td>164608</td>
          <td>108933</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>14-07-2020 15:26</td>
          <td>1.594766e+09</td>
          <td>faceUp</td>
          <td>-0.149231</td>
          <td>-0.152817</td>
          <td>-0.986572</td>
          <td>0</td>
        </tr>
      </tbody>
    </table>
    <p>164609 rows × 11 columns</p>
    </div>



.. code:: ipython3

    colx=dff.columns[7:10]
    coly=dff.columns[-1]
    colx




.. parsed-literal::

    Index(['AccelerationX', 'AccelerationY', 'AccelerationZ'], dtype='object')



.. code:: ipython3

    X=pd.DataFrame(columns=colx,data=dff)
    y=dff["Label"]
    y




.. parsed-literal::

    0         1
    1         1
    2         1
    3         1
    4         1
             ..
    164604    0
    164605    0
    164606    0
    164607    0
    164608    0
    Name: Label, Length: 164609, dtype: int64



.. code:: ipython3

    from sklearn.linear_model import LinearRegression
    import numpy as np
    from sklearn.preprocessing import StandardScaler as ss
    reg = LinearRegression()
    
    # Fit the model to the data
    
     
    # Print the coefficients of the model
    

.. code:: ipython3

    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split as tts
    X_train,X_test,y_train,y_test= tts(X,y)
    reg.fit(X_train, y_train)




.. raw:: html

    <style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">LinearRegression</label><div class="sk-toggleable__content"><pre>LinearRegression()</pre></div></div></div></div></div>



.. code:: ipython3

    y_pred=reg.predict(X_test)
    
    accuracy_score(y_test,y_pred>0.5)




.. parsed-literal::

    0.6815056010497412



.. code:: ipython3

    
    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=9)
    neigh.fit(X, y)
    
    y_pred=neigh.predict(X_test)
    
    accuracy_score(y_test,y_pred>0.5)




.. parsed-literal::

    0.8156392000583189



.. code:: ipython3

    import serial as c
    import numpy as np
    

.. code:: ipython3

    arduino_data = c.Serial('COM3', 9600)

.. code:: ipython3

    print(arduino_data.readline().strip())
    def stan(p):
        return float(p)*0.0002
    while(1):
        aa=list(map(stan,arduino_data.readline().decode("utf-8").strip().split(" ")))
        
        aa=np.array(aa)
        aa=aa.reshape(1,-1)
       
        print(aa)
        
        y_pred=neigh.predict(aa)
        if y_pred>0.5:
            print("no fall")
        else:
            print("fall")
            break
        


.. parsed-literal::

    b'MPU-6050 connection successful'
    [[ 2.1568 -1.0528  2.5696]]
    no fall
    

.. parsed-literal::

    C:\Users\selvam\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\base.py:465: UserWarning: X does not have valid feature names, but KNeighborsClassifier was fitted with feature names
      warnings.warn(
    

.. parsed-literal::

    [[ 2.1464 -1.0672  2.5272]]
    no fall
    

.. parsed-literal::

    C:\Users\selvam\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\base.py:465: UserWarning: X does not have valid feature names, but KNeighborsClassifier was fitted with feature names
      warnings.warn(
    

.. parsed-literal::

    [[ 2.1632 -1.1024  2.4968]]
    no fall
    

.. parsed-literal::

    C:\Users\selvam\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\base.py:465: UserWarning: X does not have valid feature names, but KNeighborsClassifier was fitted with feature names
      warnings.warn(
    

.. parsed-literal::

    [[ 2.2032 -1.0856  2.4528]]
    no fall
    

.. parsed-literal::

    C:\Users\selvam\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\base.py:465: UserWarning: X does not have valid feature names, but KNeighborsClassifier was fitted with feature names
      warnings.warn(
    

.. parsed-literal::

    [[ 2.26   -1.0744  2.1168]]
    no fall
    

.. parsed-literal::

    C:\Users\selvam\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\base.py:465: UserWarning: X does not have valid feature names, but KNeighborsClassifier was fitted with feature names
      warnings.warn(
    

.. parsed-literal::

    [[0. 0. 0.]]
    fall
    

.. parsed-literal::

    C:\Users\selvam\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\base.py:465: UserWarning: X does not have valid feature names, but KNeighborsClassifier was fitted with feature names
      warnings.warn(
    

