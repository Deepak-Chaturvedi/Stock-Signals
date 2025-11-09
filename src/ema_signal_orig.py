# %%
import numpy as np 
import pandas as pd 

# !pip3 install yfinance --upgrade --no-cache-dir 
# !pip3 install yfinance==0.2.61
import yfinance as yf
import datetime
# from html_table_parser.parser import HTMLTableParser
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns',120)
import datetime as dt
print(yf.__version__)
import time
# !pip3 install yfinance_cache
# import yfinance_cache  as yfc # doesn't update on new data
import ta

# %%
# get the list of NSE Listed stocks 
stocklist = pd.read_csv('D:/Work/Projects/Stock_Market/Stock_Market/data/EQUITY_L.csv')
stocklist = stocklist[['SYMBOL','NAME OF COMPANY']]
stocklist.columns = ['SYMBOL','Company name']
print(f"Total stocks {stocklist.SYMBOL.nunique()}")

## FnO stocks url  - https://chartink.com/screener/fno-stocks-all-f-o-stocks-list
fno = pd.read_csv('D:/Work/Projects/Stock_Market/Stock_Market/data/FNO Stocks - All FO Stocks List Technical Analysis Scanner.csv')
fno = fno[['Stock Name','Symbol']]
# fno.head()
# stocklist.head()

# Add .NS in Symbol names to align with yfinance API format
tickers = list(stocklist.SYMBOL.unique())
tickers = [ticker + ".NS" for ticker in tickers]
# tickers = tickers[0:5]
print(len(tickers))
print(tickers[0:5])

# %%
analysis_period = 30 # past 30 days for EMA calculation
# num_years = 365 * 1 # generate EMA-signal for past num_years YEARS
raw_data_needed = analysis_period + 365  # +1 year of buffer for 200 ema calculation
output_period = 30 # restrict output file for EMA-signals in 30 days

# %%
# Download past 'raw_data_needed' year stock data 
start = pd.Timestamp.today() - datetime.timedelta(days= raw_data_needed) # datetime.datetime(2024, 2, 1)

end = pd.Timestamp.today()  #datetime.datetime(2020, 6, 1)

start_time = time.perf_counter()

data = yf.download(tickers, start=start, end=end, threads = True)

print(f"Elapsed time: {time.perf_counter() - start_time} seconds")


# This data needs to be transposed 
df = data.stack().reset_index().rename(index=str, columns={"level_1": "Ticker"}).sort_values(['Ticker','Date'])
# df.set_index('Date', inplace=True)
# df = df.reset_index()

# Create Exponential moving averages 
df['20ema'] = df.groupby('Ticker')['Close'].transform(lambda x: x.ewm(span=20, adjust=False).mean())
df['50ema'] = df.groupby('Ticker')['Close'].transform(lambda x: x.ewm(span=50, adjust=False).mean())
df['100ema'] = df.groupby('Ticker')['Close'].transform(lambda x: x.ewm(span=100, adjust=False).mean())
df['200ema'] = df.groupby('Ticker')['Close'].transform(lambda x: x.ewm(span=200, adjust=False).mean())
df['30dv'] = df.groupby('Ticker')['Volume'].transform(lambda x: x.rolling(window=30).mean())
df['7dv'] = df.groupby('Ticker')['Volume'].transform(lambda x: x.rolling(window=7).mean())

# df['Net Volume'] = df.apply(lambda x: x['Volume'] if x['Close'] >= x['Open'] else -1 * x['Volume'], axis=1)
# df['30dvNet'] = df.groupby('Ticker')['Net Volume'].transform(lambda x: x.rolling(window=30).mean())
# df['7dvNet'] = df.groupby('Ticker')['Net Volume'].transform(lambda x: x.rolling(window=7).mean())

# # On Balance volume based on past 30 days Net Volume  
# df['obv'] = df.groupby('Ticker')['Net Volume'].transform(lambda x: x.rolling(window=30,
#                                                                             min_periods=1).sum())
df.reset_index(inplace=True)
print(df.Date.min(), df.Date.max())
df.head()


# %%
# df.to_csv('D:/Work/Projects/Stock_Market/Stock_Market/data/5YearStockData.csv', index=False)

# %% [markdown]
# ##### Find out which stocks have Golden crossovers or Death Corss in the given time period
# 

# %%
def get_ema_signal(data,start_dt, end_dt, ema1, ema2, HighSignal, LowSignal,vol_thresh):
    df_gc1 = data[(data['Date'].dt.strftime('%Y-%m-%d') >= start_dt)
                  & (data['Date'].dt.strftime('%Y-%m-%d') <= end_dt)
                  & (data[ema1] > data[ema2])]['Ticker'].drop_duplicates()
    
    df_gc2 = data[(data['Date'].dt.strftime('%Y-%m-%d') >= start_dt)
                  & (data['Date'].dt.strftime('%Y-%m-%d') <= end_dt)
                  & (data[ema1] < data[ema2])]['Ticker'].drop_duplicates()
    
    df_gc = pd.merge(left=df_gc1, right=df_gc2, how='inner', on = 'Ticker')

    
    # Find if its Golden crossover or Death crossover 
    signal_df = pd.DataFrame()
    for stock in df_gc['Ticker']:
    #     print(stock)
    #     stock = "HDFCLIFE.NS"
        stock_df = df[(df['Ticker'] == stock) & (df['Date'] >= start_dt) ]
        cross_high_index = stock_df[stock_df[ema1] > stock_df[ema2]].index.astype(int)
        cross_low_index = stock_df[stock_df[ema1] < stock_df[ema2]].index.astype(int)

        cross_index = []
        signal= []

        #Get the golden crossover dates
        for i in cross_low_index:
            for j in cross_high_index:
                if i == j - 1 :
                    cross_index.append(j)
                    signal.append(HighSignal)
        for i in cross_high_index:
            for j in cross_low_index:
                if i == j - 1 :
                    cross_index.append(j)
                    signal.append(LowSignal)

#         print(cross_index, signal)
        ticker = stock.replace('.NS','')
        for i in range(0,len(cross_index)):
            signal_df= pd.concat([signal_df,pd.DataFrame({'Symbol': ticker,
                                                          'Ticker':stock,
                                                          'signal':signal[i],
                                                          'Date': stock_df[stock_df.index==cross_index[i]]['Date'],
                                                          'Price' : stock_df[stock_df.index==cross_index[i]]['Close'],
                                                          'Volume' :  stock_df[stock_df.index==cross_index[i]]['Volume'],                                                       
                                                          '7dAvgVol' :  stock_df[stock_df.index==cross_index[i]]['7dv'] ,                                                       
                                                          '30dAvgVol' :  stock_df[stock_df.index==cross_index[i]]['30dv']                                                        
                                                         })],ignore_index=True)
            
    signal_df['7dratio'] = signal_df['Volume'] / signal_df['7dAvgVol']
    signal_df['30dratio'] = signal_df['7dAvgVol'] / signal_df['30dAvgVol']
    signal_df['volume_change'] = signal_df[['7dratio','30dratio']].max(axis=1)
    
#     print(signal_df.shape)
    #Remove low-float -  If the 30 day avg volumne is less than 1000 then remove from list
    signal_df.drop(signal_df[signal_df['30dAvgVol'] < 1000].index, inplace = True, axis = 0)
    
    #If valume is 2X the normal volume 
    signal_df['volume_signal'] = signal_df['volume_change'].apply(lambda x: 'Yes' if x> vol_thresh else 'No')
#     print(signal_df.shape)
    signal_df.drop(columns=['7dAvgVol','30dAvgVol'], inplace= True, axis= 1)
    
    signal_df = signal_df[signal_df['volume_signal'] == 'Yes']

    #Remove Penny stocks 
    signal_df = signal_df[signal_df['Price'] > 10]
    
    print(f"{signal_df.Ticker.nunique()} companies with {ema1} crossing {ema2} between {start_dt} and {end_dt}")
#           {start_dt.strftime('%Y-%m-%d')} and {end_dt.strftime('%Y-%m-%d')}")
    
    signal_df.sort_values(by=['Date','volume_change','signal'],ascending=False,inplace=True)
    return signal_df

# Start Checking from the month of March 2024  
# start_dt = datetime.datetime(2025,3,1).strftime('%Y-%m-%d')

# check only within last 30 days 
start_dt = (datetime.datetime.today() - datetime.timedelta(days= analysis_period)).strftime('%Y-%m-%d')

end_dt = pd.Timestamp.today().strftime('%Y-%m-%d') #pd #datetime.datetime(2024,4,8)
# end_dt = datetime.datetime(2020,6,1) #.strftime('%Y-%m-%d')

# Price_cross_200 = get_ema_signal(df, end_dt - dt.timedelta(days=30), end_dt, 'Close', '200ema', "Price_gt_200ema", "Price_lt_200ema", 2) #Change to 50ema for Golden cross/Death cross
Price_cross_200 = get_ema_signal(df, start_dt, end_dt, 'Close', '200ema', "Price_gt_200ema", "Price_lt_200ema", 2) #Change to 50ema for Golden cross/Death cross
golden_crossovers = get_ema_signal(df, start_dt, end_dt, '50ema', '200ema', "Golden cross", "Death cross",1.5) 

print(Price_cross_200.shape, golden_crossovers.shape)
golden_crossovers.head()

# %% [markdown]
# ##### Calculate Chande Momentum Oscillator (CMO) - 50 days
# - CMO > 0 → Bullish bias
# - CMO > 50 → Strong momentum
# - CMO < 0 → Weak or bearish
# - 📌 Look for: CMO rising and preferably > +20 at breakout
# 

# %%
def calculate_cmo(df, period,price_var):
    df['Change'] = df[price_var].diff()
    df['Gain'] = df['Change'].apply(lambda x: x if x > 0 else 0)
    df['Loss'] = df['Change'].apply(lambda x: -x if x < 0 else 0)
    
    df['Sum_Gain'] = df['Gain'].rolling(window=period).sum()
    df['Sum_Loss'] = df['Loss'].rolling(window=period).sum()
    
    df['CMO'] = 100 * (df['Sum_Gain'] - df['Sum_Loss']) / (df['Sum_Gain'] + df['Sum_Loss'])
    # 7day avg Chande momentum
    df['7dCMO'] = df.groupby('Ticker')['CMO'].transform(lambda x: x.ewm(span=7, adjust=False).mean())

    return df

chande_df= calculate_cmo(df,period=50, price_var= 'Close')
chande_df.drop(columns=['Change','Gain','Loss','Sum_Gain','Sum_Loss'],inplace=True)
# df['Symbol'] = df['Ticker'].str.replace('.NS','')
chande_df.head()

# %%
# Its possible Price crossed 200 EMA before the timeframe given here 
Price_cross_200 = pd.merge(left=Price_cross_200  , right =chande_df[['Ticker','Date','CMO']],
                             how='left', on =['Ticker','Date'] )
# Its possible Price crossed 200 EMA before the timeframe given here 
golden_crossovers = pd.merge(left=golden_crossovers  , right =chande_df[['Ticker','Date','CMO']],
                             how='left', on =['Ticker','Date'] )
print(Price_cross_200.shape,golden_crossovers.shape)
golden_crossovers.head()

# %% [markdown]
# #### Calculate Accumulation/Distribution ( Money flow Multiplier : -1 to 1  )

# %%
#Calculate the Accumulation/Distribution (AD) indicator for a given DataFrame.
def calculate_ad(df_AD):
    # Calculate the raw Accumulation/Distribution value
    df_AD['AD_raw'] = ((df_AD['Close'] - df_AD['Low']) - (df_AD['High'] - df_AD['Close'])) / (df_AD['High'] - df_AD['Low']) * df_AD['Volume']
    
    # Normalize the AD value to the range of -1 to 1
    df_AD['AD'] = round(df_AD['AD_raw'] / df_AD['Volume'],2)
    
    # Calculate the 50-day rolling sum of AD
    df_AD['AD_50'] = df_AD.groupby('Ticker')['AD_raw'].transform(lambda x: x.rolling(window=50, min_periods=1).sum())
    # Normalize AD_50 to the range of -1 to 1
    df_AD['AD_50'] = df_AD.groupby('Ticker')['AD_50'].transform(lambda x: x / x.abs().max())
    df_AD['AD_50'] = round(df_AD['AD_50'],2)

    df_AD['Symbol'] = df_AD['Ticker'].str.replace('.NS','', regex=False)

    # Drop the intermediate column
    df_AD = df_AD[['Date', 'Symbol', 'AD', 'AD_50']]

    return df_AD

df_AD = calculate_ad(df)
# print(df_AD.shape)
# df_AD.head()


# %% [markdown]
# #### Calculate the RSI-EMA(50day) indicator

# %%
# Calculate 50-day RSI and its 50-period EMA
# def RSI_Indicator(df_rsi):
#     # Use ta's EMA indicator on RSI
#     rsi_series = ta.momentum.RSIIndicator(close=df_rsi['Close'], window=50).rsi()
#     df_rsi['RSI_50'] = np.round(rsi_series,4)
#     df_rsi['RSI_EMA_50'] = np.round(ta.trend.EMAIndicator(close=rsi_series, window=50).ema_indicator(),4)

#     df_rsi['RSI_EMA_indicator'] = df_rsi['RSI_50'] - df_rsi['RSI_EMA_50']
#     # Apply sigmoid transformation
#     df_rsi['RSI_EMA_indicator'] = np.round(np.tanh(df_rsi['RSI_EMA_indicator']), 3)
#     # Positive values → RSI is above its EMA → bullish momentum
#     # Negative values → RSI is below its EMA → bearish momentum
#     # Magnitude (how close to 1 or -1) → how strong the momentum is

#     df_rsi['Symbol'] = df_rsi['Ticker'].str.replace('.NS','', regex=False)
#     df_rsi.drop(columns=['Ticker','Close'], inplace=True)

#     return df_rsi
# df_rsi = RSI_Indicator(df[['Close', 'Ticker', 'Date']].copy())
# print(df_rsi.shape)
# df_rsi.tail()

# %%
# Its possible Price crossed 200 EMA before the timeframe given here 
ema_signal = pd.merge(left=Price_cross_200  , right =golden_crossovers, how='right', on ='Symbol' , suffixes=['_1','_2'])

# ema_signal = ema_signal[ema_signal['volume_signal'] == 'Yes']
ema_signal.sort_values(by=['Date_2','volume_change_2','Date_1','volume_change_1'],ascending=(False,False,False,False)
                       ,inplace=True)
ema_signal.drop(columns=['Ticker_1','Ticker_2', 'volume_signal_1','volume_signal_2'], axis =1 , inplace=True)
ema_signal.reset_index(drop=True, inplace=True)
ema_signal.drop_duplicates(inplace= True)
print(f"{ema_signal.Symbol.nunique()} companies with crossover with Volume")
print(f"{ema_signal.shape} Total rows ")
# ema_signal.head(5)

# %% [markdown]
# The key to using the golden cross correctly—with additional filters and indicators—is to use profit targets, stop loss, and other risk management tools
# ##### CHeck list to add other supporting elements
# - Volume confirmation
# - other indicators 
# - Insider trading 

# %% [markdown]
# ##### Add the Accumulation/Distribution and RSI-EMA indicators back to dataframe

# %%
# Apply the function to calculate AD for the ema_signal dataframe
ema_signal = pd.merge(ema_signal, df_AD[['Date', 'Symbol', 'AD','AD_50']], how='left', left_on=['Date_2', 'Symbol'], right_on=['Date', 'Symbol'])
ema_signal.drop(columns=['Date'], inplace=True)

# ema_signal = pd.merge(ema_signal, df_rsi[['Date', 'Symbol', 'RSI_50','RSI_EMA_50', 'RSI_EMA_indicator']], how='left', left_on=['Date_2', 'Symbol'], right_on=['Date', 'Symbol'])
# ema_signal.drop(columns=['Date'], inplace=True)

# Drop unnecessary columns after merging
print(f"{ema_signal.shape} Total rows ")
# ema_signal.head(5)

# %% [markdown]
# ##### Get Meta information for Stocks

# %%
#Only keep required columns 
drop_columns = ['companyOfficers','compensationAsOfEpochDate','maxAge','priceHint','address1','address2','messageBoardId',
                'city','zip',	'country','website','phone','fax','industryKey','sectorKey','uuid','gmtOffSetMilliseconds',
                'timeZoneFullName','timeZoneShortName','auditRisk','boardRisk','compensationRisk','shareHolderRightsRisk',
                'overallRisk','governanceEpochDate','shareHolderRightsRisk','regularMarketPreviousClose','regularMarketOpen',
                'regularMarketDayLow','regularMarketDayHigh','ask','bid','underlyingSymbol','shortName',
                'industryDisp']


def get_stock_metadata(data,stock_column):
    # List of stock symbols
    stocks = ema_signal.Symbol.unique()

    # Define the fields you want to extract
    # fields = ["symbol", "shortName", "marketCap", "trailingPE", "sector", "industry", "fiftyTwoWeekHigh", "fiftyTwoWeekLow"]
    fields = ['symbol','currentPrice','marketCap','52WeekChange','debtToEquity', 'priceToBook','trailingPE','fiftyTwoWeekLow', 'fiftyTwoWeekHigh',
                 'trailingEps','priceToSalesTrailing12Months','PricetoCash','industry','sector','totalCashPerShare','beta']

    # 'trailingPegRatio','industry','sector','previousClose','beta','trailingPE','volume','regularMarketVolume','averageVolume',
    # 'averageVolume10days','marketCap','priceToSalesTrailing12Months','fiftyDayAverage','twoHundredDayAverage','bookValue',
    # 'priceToBook','trailingEps','symbol','longName','currentPrice','totalCashPerShare','debtToEquity','forwardPE','forwardEps',
    # 'sectorDisp','52WeekChange',

    # Initialize an empty list to store stock data
    stock_data = []
    
    # Loop through each stock symbol
    for symbol in stocks:
        try:
            # Get ticker info
            ticker = yf.Ticker(symbol+".NS")
            info = ticker.info
            
            # Extract required fields
            stock_info = {field: info.get(field, None) for field in fields}
            stock_data.append(stock_info)  # Append to the list
            time.sleep(3)  # Delay to avoid rate limiting
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
    
    # Convert list of dictionaries to DataFrame
    combined_df = pd.DataFrame(stock_data)
    combined_df.drop_duplicates(inplace=True)
    
    combined_df['SYMBOL'] =  combined_df['symbol'].str.replace('.NS', '')
    # Add the date as of which information is avilable 
    # combined_df['update date'] = yf.Ticker('RELIANCE.NS').history(period='5d').index[-1].strftime('%Y-%m-%d') #pd.Timestamp.today().strftime('%Y-%m-%d')
    combined_df['update date'] = pd.Timestamp.today().strftime('%Y-%m-%d')

    print(f"Details collected for {len(combined_df.symbol.unique())} listed companies")
    
    # Display the DataFrame
    combined_df.head()
    return combined_df

stock_meta_df = get_stock_metadata(ema_signal,'Symbol')
stock_meta_df.head()


# %%
# Convert Market cap to categories
# def get_capitalization(x):
#     x = float(x)
#     cap = ""  # Initialize cap with a default value
#     if x > 20000.0:
#         cap = "Large Cap"
#     elif x <= 20000.0 and x > 5000.00:
#         cap ="Mid Cap"
#     elif x <= 5000.0 and x > 1000.0:
#         cap ="Small Cap"
#     elif x <= 1000.0 and x > 500.0:
#         cap = "Micro Cap"
#     elif x <= 500.0:
#         cap = "Nano Cap"
#     return cap



# %%

    
# yf.Ticker('RELIANCE.NS').history(period='5d').index[-1].strftime('%Y-%m-%d') 
meta_df = stock_meta_df.iloc[:,:]
meta_df['debtToEquity'] = meta_df['debtToEquity'] /100 # Convert Debt to equity in ratio ( shold be less than 0.2 ideally)

# Clean the MCapCrore column
# meta_df['MCapCrore'] = pd.to_numeric(df['MCapCrore'], errors='coerce')
# Convert market cap to crore
meta_df['MCapCrore'] = (meta_df['marketCap'] / 1e7 ).round(2) #convert market cap in crore

meta_df['upFrom52wlow'] = (meta_df['currentPrice'] - meta_df['fiftyTwoWeekLow'])/ (
    meta_df['fiftyTwoWeekHigh']- meta_df['fiftyTwoWeekLow'])

# meta_df['Capitalization'] = meta_df['MCapCrore'].apply(lambda x : get_capitalization(x))


# Categorize the market capitalization
bins = [-float('inf'), 500, 1000, 5000, 20000, float('inf')]
labels = ["Nano Cap", "Micro Cap", "Small Cap", "Mid Cap", "Large Cap"]

meta_df['Capitalization'] = pd.cut(meta_df['MCapCrore'], bins=bins, labels=labels)

meta_df['PricetoCash'] = meta_df['currentPrice'] / meta_df['totalCashPerShare']

# meta_df.head(5)


# %%
# stock_meta_df = pd.read_csv('/kaggle/input/stock-market-datasets/NSE_comapnies_details_yfinance.csv')
# imp_vars= ['SYMBOL','currentPrice','open','dayLow','dayHigh','previousClose','volume','averageVolume','averageVolume10days',
#            'fiftyTwoWeekLow','fiftyTwoWeekHigh','priceToSalesTrailing12Months','fiftyDayAverage','twoHundredDayAverage','heldPercentInsiders',
#           'bookValue','priceToBook','beta','industry','sector','date','totalCash',	'totalCashPerShare','ebitda','totalDebt','totalRevenue','debtToEquity',
#           'date','MCapCrore','upFrom52wlow',	'Capitalization','trailingEps', 'earningsQuarterlyGrowth',
#            'trailingPE', 'heldPercentInsiders','heldPercentInstitutions','PricetoCash','update date']

join_vars = ['SYMBOL','currentPrice','Capitalization','MCapCrore','upFrom52wlow','debtToEquity', 'priceToBook','trailingPE',
             'trailingEps','priceToSalesTrailing12Months','PricetoCash','update date','industry','sector','totalCashPerShare','beta']
ema_df = pd.merge(left = ema_signal, right = meta_df[join_vars], how = 'left',left_on='Symbol', right_on='SYMBOL')

ema_df['trailingPE'] = pd.to_numeric(ema_df['trailingPE'], errors='coerce')
ema_df['priceToBook'] = pd.to_numeric(ema_df['priceToBook'], errors='coerce')

ema_df['PBXPE'] = ema_df['trailingPE'] * ema_df['priceToBook']
# ema_df['PBXPE'] = ema_df['trailingPE'] * ema_df['priceToBook']

ema_df.rename(columns={'priceToSalesTrailing12Months': 'PricetoSales'},inplace=True)

ema_df.drop(columns=['SYMBOL','Volume_1','Volume_2'],axis= 1, inplace=True)
ema_df.drop_duplicates(inplace= True)


ema_df.drop(ema_df[ema_df['trailingEps'] < 0].index, inplace=True, axis=0) # Remove loss making companies 
ema_df.drop(ema_df[ema_df['debtToEquity'] > 2.0].index, inplace=True, axis=0) # Too much debt
ema_df.drop(ema_df[ema_df['MCapCrore'] < 200].index, inplace=True, axis=0) # Market cap should be more than 200 Cr
ema_df.drop(ema_df[ema_df['trailingPE'] > 100].index, inplace=True, axis=0) # Too costly
print(ema_df.shape)
ema_df.head(2)


# %%
# ema_df.loc[:,ema_df.dtypes == float].round(2)
var1 = ['Price_1','Price_2','MCapCrore','priceToBook','trailingPE','trailingEps','PricetoCash',
        'currentPrice','PBXPE','totalCashPerShare','CMO_1','CMO_2']
var2 = ['7dratio_1','30dratio_1','volume_change_1','7dratio_2','30dratio_2','volume_change_2','PricetoSales',
       'upFrom52wlow','debtToEquity','beta']
ema_df[var1] = ema_df[var1].round(0)
ema_df[var2] = ema_df[var2].round(2)
ema_df1 = ema_df.rename(columns={'volume_change_1':'Vol1','volume_change_2':'Vol2','MCapCrore':'MCap','priceToBook':'PB',
                      'trailingPE':'PE','PricetoSales':'PS','trailingEps':'EPS',
                                'totalCashPerShare':'CashperShare','CMO_1':'Mom1',
                                'CMO_2':'Mom2'})

# Add Stock FnO flag
ema_df_2= pd.merge(left=ema_df1, right= fno['Symbol'],how='left',on='Symbol',indicator=True)
ema_df_2['FnO'] = ema_df_2['_merge'].map(lambda x: 'Yes' if x== 'both' else 'No')
ema_df_2.drop(columns=['_merge'], inplace=True)

ema_df_2.head()

# %%
# REmove this info as its not accurate completely
# industry = pd.read_csv('D:/Work/Projects/Stock_Market/Stock_Market/data/industry_averages.csv')
# industry_df = industry[['sector','industry','EPS','debtToEquity', 'PricetoSales','PricetoCash','PE','PB',
#                           'PBXPE','CashPerShare']]
# industry_df[['EPS','PricetoSales','PricetoCash','PE','PB','PBXPE','CashPerShare']] = industry_df[['EPS',
#                                                                                                   'PricetoSales','PricetoCash','PE','PB','PBXPE','CashPerShare']].round(0)
# industry_df['debtToEquity'] = industry_df[['debtToEquity']].round(2)



# industry_df.rename(columns={'EPS':'Ind EPS','debtToEquity': 'Ind debtToEquity', 
#                             'PricetoSales': 'Ind PS','PricetoCash': 'Ind PricetoCash',
#                             'PE': 'Ind PE','PB': 'Ind PB','PBXPE': 'Ind PBXPE','CashPerShare': 
#                            'Ind CashPerShare'},inplace=True)

# industry_df
# industry_df.head()

# %%
# ema_df_3= pd.merge(left=ema_df_2, right= industry_df ,how='left',on='industry',suffixes=('','_y'))
# ema_df_3.head(2)
# ema_df_3['BM_EPS'] = ema_df_3['Ind EPS'] / ema_df_3['EPS']  #Benchmarked with industry
# ema_df_3['BM_debtToEquity'] =ema_df_3['debtToEquity'] / ema_df_3['Ind debtToEquity']
# ema_df_3['BM_PricetoSales'] = ema_df_3['PS'] /ema_df_3['Ind PricetoSales'] 
# ema_df_3['BM_PricetoCash'] = ema_df_3['PricetoCash'] /ema_df_3['Ind PricetoCash'] 
# ema_df_3['BM_PE'] =ema_df_3['Ind PE'] / ema_df_3['PE']
# ema_df_3['BM_PB'] =ema_df_3['Ind PB'] / ema_df_3['PB']
# ema_df_3['BM_PBXPE'] = ema_df_3['PBXPE'] /ema_df_3['Ind PBXPE']

# ema_df_3.drop(columns=['sector_y'],inplace=True)
# ema_df_3.head()

# %%
# ema_df_3 = ema_df_2[[ 'Symbol', 'signal_1', 'Date_1', 'Price_1', '7dratio_1', '30dratio_1', 'Vol1',
#                      'Mom1','signal_2','Date_2', 'Price_2', '7dratio_2', '30dratio_2', 'Vol2','Mom2',
#                      'currentPrice','Capitalization', 'MCap', 'upFrom52wlow', 'PE', 'PB', 'PBXPE',
#                      'debtToEquity','Ind PE', 'Ind PB', 'Ind PBXPE', 'Ind debtToEquity', 'PS', 'Ind PS',
#                      'PricetoCash', 'Ind PricetoCash', 'EPS', 'Ind EPS', 'CashperShare',
#                      'Ind CashPerShare' ,'industry', 'sector', 'update date', 'FnO','beta','AD','AD_50','RSI_50','RSI_EMA_50','RSI_EMA_indicator']]

ema_df_3 = ema_df_2[[ 'Symbol', 'signal_1', 'Date_1', 'Price_1', '7dratio_1', '30dratio_1', 'Vol1',
                     'Mom1','signal_2','Date_2', 'Price_2', '7dratio_2', '30dratio_2', 'Vol2','Mom2',
                     'currentPrice','Capitalization', 'MCap', 'upFrom52wlow', 'PE', 'PB', 'PBXPE',
                     'debtToEquity', 'PS', 'PricetoCash', 'EPS', 'CashperShare',
                     'update date', 'FnO','beta','AD','AD_50']]


ema_df_3['Date_1'] = ema_df_3['Date_1'].dt.strftime('%Y-%m-%d')
ema_df_3['Date_2'] = ema_df_3['Date_2'].dt.strftime('%Y-%m-%d')
# ema_df_3.dtypes

# %%
ema_df_3.sort_values(by=['Date_2','upFrom52wlow','Vol2','Date_1','Vol1'], ascending=(False,False,False,False,False) ,inplace=True)

#keep only last 30 days data for report 
Rec30D = ema_df_3[(ema_df_3['Date_1'] >= (datetime.datetime.today() - datetime.timedelta(days= output_period)).strftime('%Y-%m-%d') ) 
                 | ( ema_df_3['Date_2'] >= (datetime.datetime.today() - datetime.timedelta(days= output_period)).strftime('%Y-%m-%d'))
                   ]
# stocks that can be shorted in option trading 
long_stocks = Rec30D[(Rec30D['signal_2'] == 'Golden cross') | (Rec30D['signal_1'] == 'Price_gt_200ema')]
short_stocks = Rec30D[(Rec30D['FnO'] == 'Yes') & ((Rec30D['signal_2'] == 'Death cross') | (Rec30D['signal_1'] == 'Price_lt_200ema'))]

#  5 Year data
# long_stocks = ema_df_3[(ema_df_3['signal_2'] == 'Golden cross') | (ema_df_3['signal_1'] == 'Price_gt_200ema')]
# short_stocks = ema_df_3[(ema_df_3['FnO'] == 'Yes') & ((ema_df_3['signal_2'] == 'Death cross') | (ema_df_3['signal_1'] == 'Price_lt_200ema'))]

long_stocks.drop(long_stocks[long_stocks['signal_2'] == 'Death cross'].index, inplace= True)
long_stocks.head()
long_stocks.head()

# %%

# stocks that can be shorted in option trading 
# long_stocks = ema_df_3[(ema_df_3['signal_2'] == 'Golden cross') | (ema_df_3['signal_1'] == 'Price_gt_200ema')]
# short_stocks = ema_df_3[(ema_df_3['FnO'] == 'Yes') & ((ema_df_3['signal_2'] == 'Death cross') | (ema_df_3['signal_1'] == 'Price_lt_200ema'))]
# long_stocks.head()

# %%
ema_df_3.to_csv(f"D:/Work/Projects/Stock_Market/Stock_Market/EMA_cross_indicator/signal_outputs/EMA_signal_stocks_with_volums_{df.Date.max().strftime('%Y-%m-%d')}.csv", index = False)
long_stocks.to_csv(f"D:/Work/Projects/Stock_Market/Stock_Market/EMA_cross_indicator/signal_outputs/long_stocks_{df.Date.max().strftime('%Y-%m-%d')}.csv", index = False)
short_stocks.to_csv(f"D:/Work/Projects/Stock_Market/Stock_Market/EMA_cross_indicator/signal_outputs/short_stocks_{df.Date.max().strftime('%Y-%m-%d')}.csv", index = False)

# %%
#Short stocks 
short_stocks.head(10)

# %%
golden_crossovers[golden_crossovers['signal'] == "Golden cross"].head(20)

# %%



