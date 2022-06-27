1) dividend yield talks about price changes directly as this can be taken as inverse price to dividend ratio
2) stock-bond correlation 
   1) Bond and equity prices reflect the discounted value of their future cash flows, where the discount rate approximately equals the sum of a:
    1) Real interest rate – compensation for the time value of money
    2) Inflation rate - compensation for the loss of purchasing power over time 
    3) Risk premium – compensation for the uncertainty of receiving future cash flows
3) 

compare the fitting vs overfitting
denoising using the autoencoder
kalman filter
seperating the signal from noise


irrespective of the activation function, applying pooling on the 

drop out layers help in creating subgraph and model averaging kind of concepts
information coefficient
bayesian inferance
prior psoterior
granger causality (https://www.analyticsvidhya.com/blog/2021/08/granger-causality-in-time-series-explained-using-chicken-and-egg-problem/)


Data Ingestion
    - Data Splitting for validation and
Data Preprocessing
    - Denoising AutoEncoder
    - Kalman Filter
    - Dimensional Reduction

Model: 
    - Training Strategies
    - Cross Validation
    - Trading strategies
          - Supply shock
          - Causality, find the casual dimension
          - 
    - LOSS Functions
    - Optimizer choice
    - Learning Rate scheduler

Plotting Data in everyphase

clustering the 

not too much position sizing and all the other volatality into consideration

    x1 | x2| x3| x4| x5| x6| x7| x8
> |Rho(t)|Inf(t)|SP5(t-5)| SP5(t-4)| SP5(t-3) | SP5(t-3) |SP5(t-1)| SP5(t)|  ---> predict SP5(t+1)


## Data Ingestion:
 -> Load Data 

## Data Preprocessing: 
-> Normalizing
-> Label encoding (Seperating Phase variables into phases)
-> Sliding Window
-> Autoencoder the Windowed Data
-> (NC) Further split the data in train 60, val 20, test 20

## Data Forecasting:
-> Prediction of S&P 500 using model variable using RNN 
-> using CNN

## Data Evaluation:
-> (NC) Predicting power test ?? (Calculate error MSE? about accuracy) using the test the model
-> (NC) Precision testing here like ROC or any other metric
-> (NC) Precision of the model against individual phase variable's phases



-> think about other Objective functions to train MSE for accuracy or ROC for precision
> after discussion, we have decided to go with accuracy, as that is most needed.
> 

## Trading Strategy
MSE : 0.93

PredictedValue (-+) (0.07)^2 -->  new Predicted value 
-> (NC) Trading on Direction (100% in or 100% out) also apply phase variable
> We have 1 month invest 1 in the 
> We have 5 timestamps we predict the direction to enter or not, find the return on the invest on every recurring timelag and take the position, if direction is +ve, invest and if direction is -ve , we pose the P&L.
> We need to fetch the sharpe ratio of the model which evaluates the model strategy.
> 100% in or 100% out, is not it the plain sharpe ratio of S&P 500.


## experiment segLearn

<!-- # Split the data into train test and validation using the windowgenerator as we want the time series methodology while doing this also refer the documents
# perform pre-classification using the label encoder or any other logic like logistic regression or any classification logic on three signals CPI, Rho, DIL
# perform Dimensional Reduction
#     compare the fitting vs overfitting
# denoising using the autoencoder
# kalman filter
# seperating the signal from noise


# irrespective of the activation function, applying pooling on the

# drop out layers help in creating subgraph and model averaging kind of concepts
# information coefficient
# bayesian inferance
# prior psoterior
# granger causality


# Data Ingestion
#     - Data Splitting for validation and
# Data Preprocessing
#     - Denoising AutoEncoder
#     - Kalman Filter
#     - Dimensional Reduction

# Model:
#     - Training Strategies
#     - Cross Validation
#     - Trading strategies
#           - Supply shock
#           - Causality, find the casual dimension
#           -
#     - LOSS Functions
#     - Optimizer choice
#     - Learning Rate scheduler

# Plotting Data in everyphase

# clustering the -->