import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# Read the CSV file 
df_HP = pd.read_csv('houseprice_UK.csv')
df_RTR = pd.read_csv('RTR.csv')
df_1MLIBOR = pd.read_csv('1MLIBOR.csv')
df_DCC = pd.read_csv('DCC_data.csv')

# Convert 'Date' column to datetime format
df_RTR['Date'] = pd.to_datetime(df_RTR['Date'])
df_HP['Date'] = pd.to_datetime(df_HP['Date'])
df_1MLIBOR['Date'] = pd.to_datetime(df_1MLIBOR['Date'])
df_DCC['Date'] = pd.to_datetime(df_DCC['Date'], dayfirst= True)

# Calculate the percentage change in 'RTR Rate'
df_RTR['Return'] = df_RTR['RTR Rate'].pct_change() * 100

df_1MLIBOR["Future Quote"] = 100 - df_1MLIBOR['1M LIBOR Rate']
df_1MLIBOR['Return'] = df_1MLIBOR['Future Quote'].pct_change() * 100

# Filter data from '1995-02-28' to '1995-06-30'
start_date = '1995-02-28'
end_date = '2023-03-31'

df_1MLIBOR = df_1MLIBOR[(df_1MLIBOR['Date'] >= start_date) & (df_1MLIBOR['Date'] <= end_date)].reset_index(drop=True)
df_RTR = df_RTR[(df_RTR['Date'] >= start_date) & (df_RTR['Date'] <= end_date)].reset_index(drop=True)
df_HP = df_HP[(df_HP['Date'] >= '1995-02-01') & (df_HP['Date'] <= end_date)].reset_index(drop=True)
df_DCC = df_DCC[(df_DCC['Date'] >= start_date) & (df_DCC['Date'] <= end_date)].reset_index(drop=True)


house_price = df_HP["House Price"]

RTR_monthly_rate = df_RTR['RTR Rate']/100/12

date = df_RTR["Date"]

loans = house_price * 0.75 

future_quote = df_1MLIBOR["Future Quote"]
forcasted_correlation = df_DCC["Forcasted Correlation"]
forcasted_volatility_RTR = df_DCC["Forcasted Volatility RTR"]
forcasted_volatility_future = df_DCC["Forcasted Volatility Future"]

  

loan = loans[0]
length = len(loans)

# Assuming 'interests' is your list or array causing the IndexError

def hedging_loan(loan, length, start_index, treshold):
        
    loan_progression = np.zeros(length)
    interests = np.zeros(length)
    monthly_payments = np.zeros(length)
    
    #Hedge journey
    interests_delta = np.zeros(length)
    forcasted_interest_volatility = np.zeros(length)
        #DCC
    DCC_OHR = np.zeros(length)
    DCC_delta = np.zeros(length)
    DCC_O_delta = np.zeros(length)
        #CCC
    constant_correlation = -0.421639605
    CCC_OHR = np.zeros(length)
    CCC_delta = np.zeros(length)
        #RG
    OHR_RG = -4.875054646
    RG_delta = np.zeros(length)
    RGO_delta = np.zeros(length)
    
    #RG and DCCO
    RG_DCCO_delta = np.zeros(length)
    
    #threshold_value
    delta_int_os = []
    delta_DCC_os = []
    delta_DCCO_os = []
    delta_CCC_os = []
    delta_RG_os = []
    delta_RGO_os = []
    delta_RG_DCCO_os = []
    
        
    for i in range(length):
        if i == 0:
            
            loan_progression[i] = loan
            interests[i] = loan * RTR_monthly_rate[i+start_index]
            
            loan_term = 360
            monthly_payments[i]= loan*((RTR_monthly_rate[i+start_index]*((RTR_monthly_rate[i+start_index]+1)**loan_term))/(((1+RTR_monthly_rate[i+start_index])**loan_term)-1))
            
            interests_delta[i] = 0
            forcasted_interest_volatility[i] = 0
            DCC_OHR[i] = 0
            DCC_delta[i] = 0
            CCC_OHR[i] = 0
            CCC_delta[i] = 0
            RG_delta[i] = 0
            DCC_O_delta[i] = 0

            
        else:
            refund = monthly_payments[i-1] - interests[i-1]
            loan_progression[i] = loan_progression[i-1] - refund
            interests[i] = loan_progression[i] * RTR_monthly_rate[i+start_index]
            
            
            loan_term = 360 - i
            monthly_payments[i]= loan_progression[i]*((RTR_monthly_rate[i+start_index]*((RTR_monthly_rate[i+start_index]+1)**loan_term))/(((1+RTR_monthly_rate[i+start_index])**loan_term)-1))
            
            #Hedge
                
            #interest cash flow & volatility
            interests_delta[i] = interests[i-1] - interests[i]
            interest_realised_volatility = abs(interests_delta[i]/interests[i-1])
            
            
   
            forcasted_interest_volatility[i] = abs(forcasted_volatility_RTR[i+start_index] * (loan_progression[i]/loan_progression[i-1]) - refund/loan_progression[i-1])
          
                #OHR
            DCC_OHR[i] = forcasted_correlation[i+start_index] * (forcasted_interest_volatility[i]/forcasted_volatility_future[i+start_index])
            

            CCC_OHR[i] = constant_correlation * (forcasted_interest_volatility[i]/forcasted_volatility_future[i+start_index])
           
                
                #Contracts to order
            DCC_nb_contract = round(DCC_OHR[i] * (interests[i-1]/future_quote[i+ start_index-1]))
            
            if forcasted_correlation[i+start_index] < 0:
                DCC_nb_contract_O = round(DCC_OHR[i] * (interests[i-1]/future_quote[i+start_index-1]))
                RGO_nb_contract = round(OHR_RG * (interests[i-1]/future_quote[i+start_index-1]))
                
                
                
            else:
                DCC_nb_contract_O = 0
                RGO_nb_contract = 0
                
           
                
            CCC_nb_contract = round(CCC_OHR[i] * (interests[i-1]/future_quote[i+start_index-1]))
            RG_nb_contract = round(OHR_RG * (interests[i-1]/future_quote[i+start_index-1]))
           
                
                #Future cash flow (cf)
            DCC_cf = DCC_nb_contract * (future_quote[i + start_index]-future_quote[i-1+ start_index])
            
            DCC_cfO = DCC_nb_contract_O * (future_quote[i+ start_index]-future_quote[i-1+ start_index])
            
            CCC_cf = CCC_nb_contract * (future_quote[i+ start_index]-future_quote[i-1+ start_index])
            RG_cf = RG_nb_contract * (future_quote[i+ start_index]-future_quote[i-1+ start_index])
            
            RGO_cf = RGO_nb_contract * (future_quote[i+ start_index]-future_quote[i-1+ start_index])
         
                #Hedge cash flow (cf)
            DCC_delta[i] = interests_delta[i] + DCC_cf
            DCC_O_delta[i] = interests_delta[i] + DCC_cfO
            CCC_delta[i] = interests_delta[i] + CCC_cf
            RG_delta[i] = interests_delta[i] + RG_cf
            RGO_delta[i] = interests_delta[i] + RGO_cf
            
            if forcasted_correlation[i+start_index] < 0:
                RG_DCCO_delta[i] = interests_delta[i] + DCC_cfO
            else:
                RG_DCCO_delta[i] = interests_delta[i] + RG_cf
            
            
            # But the sign "<" for in sample and ">" for out sample !
            if interest_realised_volatility > treshold:
                #if  i + start_index < 257 or i + start_index > 289:
                
                delta_int_os.append(interests_delta[i])
                delta_DCC_os.append(DCC_delta[i])
                delta_DCCO_os.append(DCC_O_delta[i])
                delta_CCC_os.append(CCC_delta[i])
                delta_RG_os.append(RG_delta[i])
                delta_RGO_os.append(RGO_delta[i])
                delta_RG_DCCO_os.append(RG_DCCO_delta[i])
                
    
    var__int = np.var(delta_int_os)
    var__DCC = np.var(delta_DCC_os)
    var__DCCO = np.var(delta_DCCO_os)
    var__CCC = np.var(delta_CCC_os)
    var__RG = np.var(delta_RG_os)
    var__RGO = np.var(delta_RGO_os)
    var__RG_DCCO = np.var(delta_RG_DCCO_os)
       
    return var__int, var__DCC , var__DCCO, var__CCC, var__RG, len(delta_int_os), var__RGO, var__RG_DCCO

###################################################################################
#Finding the fitted function weighted

var___interest = []
var___DCCO = []
var___CCC = []
var___RG = []
length___var = []
var___RGO = []
var___RG_DCCO = []
    
for i in range(len(loans)):
    
    threshold = 0
     
    loan = loans[i]
        
    length = len(loans) - i
    start_index = i
    
    variance_interest, variance_DCC, variance_DCCO, variance_CCC, variance_RG, lengthvar_, variance_RGO, variance_RG_DCCO = hedging_loan(loan, length, start_index, threshold)
    
    var___interest.append(variance_interest)
    var___DCCO.append(variance_DCCO)
    var___CCC.append(variance_CCC)
    var___RG.append(variance_RG)
    length___var.append(lengthvar_)
    var___RGO.append(variance_RGO)
    var___RG_DCCO.append(variance_RG_DCCO)
        
# Create a DataFrame from lists
df = pd.DataFrame({
    'variance interest': var___interest,
    'variance DCCO': var___DCCO,
    'variance CCC': var___CCC,
    'variance RG': var___RG,
    'length': length___var
    
    })



    
lengthvar = df["length"].values

    # Sort the DataFrame based on a specific column in descending order
sorted_df = df.sort_values(by='length', ascending=False)


    # Group by column 'A' and calculate the mean of other columns
grouped_df = df.groupby('length').mean().reset_index()


    # Sort the DataFrame based on a specific column in descending order
sorted_df = grouped_df.sort_values(by='length', ascending=False)

    # Reset the index
sorted_df = sorted_df.reset_index(drop=True)


sorted_df = sorted_df[:-2]

 
sorted_df["info loss"] = sorted_df["length"].pct_change()

    

    #reterive columns and put them in an array
varint = sorted_df["variance interest"].values
length = sorted_df["length"].values

plt.figure(1)
plt.scatter(length, varint, label="Interest Variance", marker='o')
        #plt.scatter(up_RTR, up_utility, label="Utility Returns", marker='+')

plt.xlabel('Observations')
plt.ylabel('Variance')
plt.gca().invert_xaxis()
plt.axhline(0, color='black', linewidth=0.5)
plt.title('Variance Relative to Observations')
plt.legend()
plt.show()


x = np.arange(1, len(length[:-117]) + 1)
#match y length

y_ = np.concatenate((varint[:90], varint[170:]))
#delete 2008 unsual volatilities and other values

y = y_[:-37]

# Define the quadratic function
def quadratic_function(x, a, b, c ):
    return a * x ** 3  + b * x + c

# Curve fitting to find coefficients for the quadratic function
popt, pcov = curve_fit(quadratic_function, x, y)

print(popt)
a, b, c = popt
#Create a polynomial function using the fitted coefficients
#fitted_function = lambda x: popt[0] * x ** popt[1] + popt[2] * x ** popt[3] + popt[4] * x + popt[5]
fitted_function = quadratic_function(x, a,b,c)


gap = (max(fitted_function) - fitted_function)
weight = gap/np.sum(gap)

plt.figure(2)

# Plot the original data and the fitted function
plt.scatter(x, y, label='Original data')
plt.plot(x, fitted_function, 'r-', label='Fitted function')
#plt.plot(x, weight, 'r-', label='weight')

plt.legend()
plt.xlabel('x')
plt.ylabel('y')

plt.show()



##################################################################################
#process 

thresholds = np.linspace(0.0,0.06, 50)

WALG_RG_DCCOs = []
WALG_RG_RGOs = []
WALG_RG_RG_DCCOs = []
volatilities = []
sum_lengthvars = []
avg_rate_loss_infos = []

WALG_DCCO_ = []
WALG_RG_ = []

corr = []

for threshold in thresholds:
     
    var__interest = []
    var__DCCO = []
    var__CCC = []
    var__RG = []
    length__var = []
    var__RGO = []
    var__RG_DCCO = []
    
    RTR_returns = []
    future_returns = []
    

    for i in range(len(loans)):
     
        loan = loans[i]
        
        length = len(loans) - i
        start_index = i
    
        variance_interest, variance_DCC, variance_DCCO, variance_CCC, variance_RG, lengthvar_, variance_RGO, variance_RG_DCCO = hedging_loan(loan, length, start_index, threshold)
    
        var__interest.append(variance_interest)
        var__DCCO.append(variance_DCCO)
        var__CCC.append(variance_CCC)
        var__RG.append(variance_RG)
        length__var.append(lengthvar_)
        var__RGO.append(variance_RGO)
        var__RG_DCCO.append(variance_RG_DCCO)
        
        if i == 0:
            
            RTR_return = 0
            future_return = 0
        
        else:
            RTR_return = (RTR_monthly_rate[i]-RTR_monthly_rate[i-1])/RTR_monthly_rate[i-1]
            future_return = (future_quote[i]-future_quote[i-1])/future_quote[i-1]
            
        if abs(RTR_return) > threshold:
            RTR_returns.append(RTR_return)
            future_returns.append(future_return)
            
     
    # Calculate correlation coefficient
    correlation_matrix = np.corrcoef(RTR_returns, future_returns)
    correlation_coefficient = correlation_matrix[0, 1]
    corr.append(correlation_coefficient)
 
       
    # Create a DataFrame from lists
    df = pd.DataFrame({
        'variance interest': var__interest,
        'variance DCCO': var__DCCO,
        'variance CCC': var__CCC,
        'variance RG': var__RG,
        'length': length__var
    
    })
    
    df_O = pd.DataFrame({
        'variance interest': var__interest,
        'variance RGO': var__RGO,
        'variance RG DCCO': var__RG_DCCO,
        'variance DCCO': var__DCCO,
        'length': length__var
    
    })
    
    # Sort the DataFrame based on a specific column in descending order
    sorted_df = df.sort_values(by='length', ascending=False)
    sorted_dfO = df_O.sort_values(by='length', ascending=False)

    # Group by column 'A' and calculate the mean of other columns
    grouped_df = df.groupby('length').mean().reset_index()
    grouped_dfO = df_O.groupby('length').mean().reset_index()

    # Sort the DataFrame based on a specific column in descending order
    sorted_df = grouped_df.sort_values(by='length', ascending=False)
    sorted_dfO = grouped_dfO.sort_values(by='length', ascending=False)
    # Reset the index
    sorted_df = sorted_df.reset_index(drop=True)
    sorted_dfO = sorted_dfO.reset_index(drop=True)

    sorted_df = sorted_df[:-2]
    sorted_dfO = sorted_dfO[:-2] 
 
    sorted_df["info loss"] = sorted_df["length"].pct_change()
    sorted_dfO["info loss"] = sorted_dfO["length"].pct_change() 
   
    #reterive columns and put them in an array
    var_RG = sorted_df["variance RG"].values
    var_DCCO = sorted_df["variance DCCO"].values
    var_RGO = sorted_dfO["variance RGO"].values
    var_RG_DCCO = sorted_dfO["variance RG DCCO"].values
    var_int = sorted_df["variance interest"].values
    length_var = sorted_df["length"].values
    
    #fit function to length
    x = np.arange(1, len(length_var)+1)
    y = quadratic_function(x, a, b, c )
    
   
    gap = (max(y) - y)
    weight = gap/np.sum(gap)

    
    #length_var decline expentionnaly, assume loss of information increase at the fitted function rate
    exp_decline_length = length_var * weight
    
    #caluclate perc gap
    gap_RG_DCCO = (var_RG - var_DCCO)/var_RG  * 100
    gap_RG_RGO = (var_int - var_RGO)/var_int * 100
    gap_RG_RG_DCCO =(var_RG - var_RG_DCCO)/var_RG  * 100
    
    gap_int_DCCO = (var_int - var_DCCO)/var_int * 100
    gap_int_RG = (var_int - var_RG)/var_int * 100
    
    #weighted perc gap
    weighted_gap_RG_DCCO = gap_RG_DCCO * exp_decline_length/sum(exp_decline_length)
    weighted_gap_RG_RGO = gap_RG_RGO * exp_decline_length/sum(exp_decline_length)
    weighted_gap_RG_RG_DCCO = gap_RG_RG_DCCO * exp_decline_length/sum(exp_decline_length)
    
    weighted_gap_RG = gap_int_RG * exp_decline_length/sum(exp_decline_length)
    weighted_gap_DCCO = gap_int_DCCO * exp_decline_length/sum(exp_decline_length)
    
    WALG_RG_DCCO = np.sum(weighted_gap_RG_DCCO)
    WALG_RG_RGO =  np.sum(weighted_gap_RG_RGO)
    WALG_RG_RG_DCCO =  np.sum(weighted_gap_RG_RG_DCCO)
    
    WALG_DCCO = np.sum(weighted_gap_DCCO)
    WALG_RG = np.sum(weighted_gap_RG)
    
    WALG_RG_DCCOs.append(WALG_RG_DCCO)
    WALG_RG_RGOs.append(WALG_RG_RGO)
    WALG_RG_RG_DCCOs.append(WALG_RG_RG_DCCO)
    volatilities.append(threshold*100)
    sum_lengthvars.append(sum(length_var))
    
    WALG_DCCO_.append(WALG_DCCO)
    WALG_RG_.append(WALG_RG)

 
    
# Create a DataFrame from lists
df = pd.DataFrame({
    'treshold volatility (%)': volatilities,
    'weighted avg gap var DCCO': WALG_DCCO_,
    'weighted avg gap var RG': WALG_RG_,
    'correlation': corr,
    'info': sum_lengthvars
    })

df["info loss (%)"] = df["info"].pct_change() * 100
    
print(df) 
        
vol = df["treshold volatility (%)"].values
wagvDCCO = df["weighted avg gap var DCCO"].values
wagvRG = df["weighted avg gap var RG"].values



plt.plot(vol, wagvDCCO, 'b-', label='gap Int-DCCO (%)')
plt.plot(vol, wagvRG, 'r-', label='gap Int-RG(%)')


#plt.axhline(0, color='black',linewidth=0.5)
plt.grid(True)

plt.legend()
plt.xlabel('Volatility (%)')
plt.ylabel('WAPG RG DCCO (%) ')

plt.figure()

plt.plot(vol, corr, 'y-', label = "Correlation" )
plt.xlabel('Volatility')
plt.ylabel('Correlation')

plt.legend()
plt.grid(True)

plt.show()

     
        
        
    
    
    
   
    
    
    

   
    




