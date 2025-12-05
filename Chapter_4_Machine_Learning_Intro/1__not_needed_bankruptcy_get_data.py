"""
Converted from: Chapter 4 - 1 _NOT NEEDED_Bankruptcy_Get_Data.ipynb

This script contains code extracted from a Jupyter notebook.
Code has been organized for modular execution.
"""

import pandas as pd
import requests
import json
import os
import numpy as np
from matplotlib import pyplot as plt

"""
Bankruptcy Prediction Toy Example
"""
#
# The data is just a toy example, tieing into Altmans paper, so it is very rough.
#
# The way the data for bankrupt companies was actually obtained was a combination of using EODHistoricalData and manually reading the financial statements.
#
# For Non-bankrupt companies, the StockData.csv file was used to gather a random sampling of american companies.
#
# Some parts from later in the green book are required, such as StockData.csv and the API interaction to fetch company data. This file is left here for readers to tinker. There's no need for it as the data is already provided for the toy example.
#
#
# For list of bankrupt companies using:
# https://en.wikipedia.org/wiki/Category:Companies_that_filed_for_Chapter_11_bankruptcy_in_2020
# Not a perfect list, but is more than enough for our toy example.
# Take the variables roughly a year before bankruptcy, or closer.
#
#
#
"""
Obtaining Stock Financial Data
"""

myToken = '12345abc.12345xyz' # Own token from EODHistoricalData

# Constructing our DataFrame first with dummy figures. Links to filings are here to see the figures firsthand.

badStocks = pd.DataFrame(columns=['Name', 'Ticker', 'YearBankrupt', '(CA-CL)/TA', 'RE/TA', 'EBIT/TA', 'BookEquity/TL'])
#https://www.sec.gov/ix?doc=/Archives/edgar/data/0001657853/000165785320000007/hghthc201910-k.htm
a = {'Name':'The Hertz Corporation', 'Ticker':'HTZ', 'YearBankrupt':2020, '(CA-CL)/TA':2, 'RE/TA':2, 'EBIT/TA':2, 'BookEquity/TL':2}

b = {'Name':'JCPenney', 'Ticker':'JCP', 'YearBankrupt':2020, '(CA-CL)/TA':2, 'RE/TA':2, 'EBIT/TA':2,  'BookEquity/TL':2}

c = {'Name':'Tailored Brands', 'Ticker':'TLRD', 'YearBankrupt':2020, '(CA-CL)/TA':2, 'RE/TA':2, 'EBIT/TA':2, 'BookEquity/TL':2}

#https://www.sec.gov/ix?doc=/Archives/edgar/data/314808/000031480820000015/val-20191231x10k.htm#s06E3990195275D52901F43A8F09A3743
d = {'Name':'Valaris plc', 'Ticker':'VALP', 'YearBankrupt':2020, '(CA-CL)/TA':2, 'RE/TA':2, 'EBIT/TA':2, 'BookEquity/TL':2}

#http://investors.chk.com/sec-filings?cat=1
e = {'Name':'Chesapeake Energy Corporation', 'Ticker':'CHK', 'YearBankrupt':2020, '(CA-CL)/TA':2, 'RE/TA':2, 'EBIT/TA':2, 'BookEquity/TL':2}

#https://investors.intelsat.com/financial-information/sec-filings?field_nir_sec_form_group_target_id%5B%5D=471&field_nir_sec_date_filed_value=&items_per_page=10#views-exposed-form-widget-sec-filings-table#views-exposed-form-widget-sec-filings-table
f = {'Name':'Intelsat Corporation', 'Ticker':'INTE', 'YearBankrupt':2020, '(CA-CL)/TA':2, 'RE/TA':2, 'EBIT/TA':2, 'BookEquity/TL':2}

#https://www.sec.gov/edgar/browse/?CIK=14195
g = {'Name':'Briggs & Stratton', 'Ticker':'BGGS', 'YearBankrupt':2020, '(CA-CL)/TA':2, 'RE/TA':2, 'EBIT/TA':2, 'BookEquity/TL':2}

#https://last10k.com/sec-filings/akrx#link_fullReport
h = {'Name':'Akorn Inc.', 'Ticker':'AKRX', 'YearBankrupt':2020, '(CA-CL)/TA':2, 'RE/TA':2, 'EBIT/TA':2, 'BookEquity/TL':2}

#http://investors.mcclatchy.com/sec-filings?field_nir_sec_form_group_target_id%5B%5D=471&field_nir_sec_date_filed_value=#views-exposed-form-widget-sec-filings-table
i = {'Name':'McClatchy', 'Ticker':'MNI', 'YearBankrupt':2020, '(CA-CL)/TA':2, 'RE/TA':2, 'EBIT/TA':2, 'BookEquity/TL':2}

#https://last10k.com/sec-filings/lby#link_fullReport
j = {'Name':'Libbey Incorporated', 'Ticker':'LBY', 'YearBankrupt':2020, '(CA-CL)/TA':2, 'RE/TA':2, 'EBIT/TA':2, 'BookEquity/TL':2}



#https://www.sec.gov/edgar/browse/?CIK=1282266
k = {'Name':'Windstream Holdings', 'Ticker':'WINM', 'YearBankrupt':2019, '(CA-CL)/TA':2, 'RE/TA':2, 'EBIT/TA':2, 'BookEquity/TL':2}

#https://www.sec.gov/edgar/browse/?CIK=1282266
l = {'Name':'Petroleum Helicopters International', 'Ticker':'PHIIK', 'YearBankrupt':2019, '(CA-CL)/TA':2, 'RE/TA':2, 'EBIT/TA':2, 'BookEquity/TL':2}

#https://www.sec.gov/edgar/browse/?CIK=1265572&owner=exclude
m = {'Name':'Kona Grill Inc.', 'Ticker':'KONA', 'YearBankrupt':2019, '(CA-CL)/TA':2, 'RE/TA':2, 'EBIT/TA':2, 'BookEquity/TL':2}

#https://www.sec.gov/edgar/browse/?CIK=1409532
n = {'Name':'Insys Therapeutics', 'Ticker':'INSY', 'YearBankrupt':2019, '(CA-CL)/TA':2, 'RE/TA':2, 'EBIT/TA':2, 'BookEquity/TL':2}

#https://www.sec.gov/edgar/browse/?CIK=724571
o = {'Name':'Freds Inc.', 'Ticker':'FRED', 'YearBankrupt':2019, '(CA-CL)/TA':2, 'RE/TA':2, 'EBIT/TA':2, 'BookEquity/TL':2}

#https://www.sec.gov/edgar/browse/?CIK=931336&owner=exclude
p = {'Name':'DEAN FOODS CO', 'Ticker':'DFOD', 'YearBankrupt':2019, '(CA-CL)/TA':2, 'RE/TA':2, 'EBIT/TA':2, 'BookEquity/TL':2}

#https://www.sec.gov/edgar/browse/?CIK=865941
q = {'Name':'Celadon Group', 'Ticker':'CGIP', 'YearBankrupt':2019, '(CA-CL)/TA':2, 'RE/TA':2, 'EBIT/TA':2, 'BookEquity/TL':2}



#https://www.sec.gov/edgar/browse/?CIK=1310067&owner=exclude
r = {'Name':'Sears Holdings Corporation', 'Ticker':'SHLD', 'YearBankrupt':2018, '(CA-CL)/TA':2, 'RE/TA':2, 'EBIT/TA':2, 'BookEquity/TL':2}

#https://www.sec.gov/edgar/browse/?CIK=1400891&owner=exclude
s = {'Name':'iHeartMedia', 'Ticker':'IHRT', 'YearBankrupt':2018, '(CA-CL)/TA':2, 'RE/TA':2, 'EBIT/TA':2, 'BookEquity/TL':2}

#https://www.sec.gov/edgar/browse/?CIK=920321
t = {'Name':'Cenveo', 'Ticker':'CVO', 'YearBankrupt':2018, '(CA-CL)/TA':2, 'RE/TA':2, 'EBIT/TA':2, 'BookEquity/TL':2}

#https://www.sec.gov/edgar/browse/?CIK=878079
u = {'Name':'Bon-Ton Holdings, Inc.', 'Ticker':'BONT', 'YearBankrupt':2018, '(CA-CL)/TA':2, 'RE/TA':2, 'EBIT/TA':2, 'BookEquity/TL':2}



#https://www.sec.gov/edgar/browse/?CIK=1396279
v = {'Name':'H. H. Gregg Inc.', 'Ticker':'HGGG', 'YearBankrupt':2017, '(CA-CL)/TA':2, 'RE/TA':2, 'EBIT/TA':2, 'BookEquity/TL':2}

#https://www.sec.gov/edgar/browse/?CIK=1058623&owner=exclude
w = {'Name':'Cumulus Media, Inc.', 'Ticker':'CMLS', 'YearBankrupt':2017, '(CA-CL)/TA':2, 'RE/TA':2, 'EBIT/TA':2, 'BookEquity/TL':2}

#https://www.sec.gov/edgar/browse/?CIK=1471261
x = {'Name':'Cobalt International Energy, Inc.', 'Ticker':'CIEI', 'YearBankrupt':2017, '(CA-CL)/TA':2, 'RE/TA':2, 'EBIT/TA':2, 'BookEquity/TL':2}



#https://www.sec.gov/edgar/browse/?CIK=1054579
y = {'Name':'Hastings Entertainment', 'Ticker':'HAST', 'YearBankrupt':2016, '(CA-CL)/TA':2, 'RE/TA':2, 'EBIT/TA':2, 'BookEquity/TL':2}

#https://www.sec.gov/edgar/browse/?CIK=1064728&owner=exclude
z = {'Name':'Peabody Energy', 'Ticker':'BTU', 'YearBankrupt':2016, '(CA-CL)/TA':2, 'RE/TA':2, 'EBIT/TA':2, 'BookEquity/TL':2}

badStocks=badStocks.append([a,b,c,d,e,f,g,h,i,j,
                 k,l,m,n,o,p,q,
                 r,s,t,u,
                 v,w,x,
                 y,z], ignore_index=True)
badStocks

"""
Bankrupt Stocks
"""
"""
Get Fundamental Data From EODHistoricalData (When Available)
"""
# When data isn't available, the data was filled in by hand. See the badStocks_building.csv file.

listOfStocks = list(badStocks['Ticker']+'.US')

# Some tickers have issues.
del(listOfStocks[3])
del(listOfStocks[4])
del(listOfStocks[4])
del(listOfStocks[7])
del(listOfStocks[11])
del(listOfStocks[12])
del(listOfStocks[17])

listOfStocks

session = requests.Session()
params = {'api_token': myToken}
r = session.get('https://eodhistoricaldata.com/api/user/', params=params)
r.text


def get_fundamental_data(symbol='MCD.US', api_token='XXXXXXXXXXXXXX', session=None): #Move to myFunctions.py
    '''Get fundamental data as a dictionary.'''
    if session is None:
        session = requests.Session()
    url = 'https://eodhistoricaldata.com/api/fundamentals/'+ symbol
    params = {'api_token': api_token}
    r = session.get(url, params=params)
        
    if r.status_code == requests.codes.ok:
        return json.loads(r.text) # returns a dictionary
    else:
        raise Exception(r.status_code, r.reason, url)

singleStockData = get_fundamental_data('HTZ.US', myToken)
singleStockData.keys() # view the keys

singleStockData['Financials']['Balance_Sheet']['yearly']['2020-12-31']

singleStockData['Financials']['Income_Statement']['yearly']['2020-12-31']

(float(singleStockData['Financials']['Balance_Sheet']['yearly']['2020-12-31']['totalCurrentAssets']) - \
float(singleStockData['Financials']['Balance_Sheet']['yearly']['2020-12-31']['totalCurrentLiabilities']))/\
float(singleStockData['Financials']['Balance_Sheet']['yearly']['2020-12-31']['totalAssets'])

float(singleStockData['Financials']['Balance_Sheet']['yearly']['2020-12-31']['retainedEarnings'])/\
float(singleStockData['Financials']['Balance_Sheet']['yearly']['2020-12-31']['totalAssets'])

singleStockData['Financials']['Balance_Sheet']['yearly']['2020-12-31']['totalCurrentAssets']

singleStockData = get_fundamental_data('BTU.US', myToken) # issues:VALP, INTE, BGGS, WINM, DFOD, SHLD, CIEI
singleStockData['General'] # view the keys

# dataframe columns
cols = ['Code',
'Type',
'Name',
'Exchange',
'CurrencyCode',
'CurrencyName',
'CurrencySymbol',
'CountryName',
'CountryISO',
'ISIN',
'CUSIP',
'CIK',
'EmployerIdNumber',
'FiscalYearEnd',
'IPODate',
'InternationalDomestic',
'Sector',
'Industry',
'GicSector',
'GicGroup',
'GicIndustry',
'GicSubIndustry',
'HomeCategory',
'IsDelisted',
#'Description',
'Address',
'AddressData',
'Listings',
#'Officers',
'Phone',
'WebURL',
'LogoURL',
'FullTimeEmployees',
'UpdatedAt',
'SharesOutstanding',
'SharesFloat',
'PercentInsiders',
'PercentInstitutions',
'SharesShort',
'SharesShortPriorMonth',
'ShortRatio',
'ShortPercentOutstanding',
'ShortPercentFloat',
'date',
'filing_date',
'currency_symbol',
'totalAssets',
'intangibleAssets',
'earningAssets',
'otherCurrentAssets',
'totalLiab',
'totalStockholderEquity',
'deferredLongTermLiab',
'otherCurrentLiab',
'commonStock',
'retainedEarnings',
'otherLiab',
'goodWill',
'otherAssets',
'cash',
'totalCurrentLiabilities',
'shortTermDebt',
'shortLongTermDebt',
'shortLongTermDebtTotal',
'otherStockholderEquity',
'propertyPlantEquipment',
'totalCurrentAssets',
'longTermInvestments',
'netTangibleAssets',
'shortTermInvestments',
'netReceivables',
'longTermDebt',
'inventory',
'accountsPayable',
'totalPermanentEquity',
'noncontrollingInterestInConsolidatedEntity',
'temporaryEquityRedeemableNoncontrollingInterests',
'accumulatedOtherComprehensiveIncome',
'additionalPaidInCapital',
'commonStockTotalEquity',
'preferredStockTotalEquity',
'retainedEarningsTotalEquity',
'treasuryStock',
'accumulatedAmortization',
'nonCurrrentAssetsOther',
'deferredLongTermAssetCharges',
'nonCurrentAssetsTotal',
'capitalLeaseObligations',
'longTermDebtTotal',
'nonCurrentLiabilitiesOther',
'nonCurrentLiabilitiesTotal',
'negativeGoodwill',
'warrants',
'preferredStockRedeemable',
'capitalSurpluse',
'liabilitiesAndStockholdersEquity',
'cashAndShortTermInvestments',
'propertyPlantAndEquipmentGross',
'accumulatedDepreciation',
'commonStockSharesOutstanding',
'investments',
'changeToLiabilities',
'totalCashflowsFromInvestingActivities',
'netBorrowings',
'totalCashFromFinancingActivities',
'changeToOperatingActivities',
'netIncome',
'changeInCash',
'totalCashFromOperatingActivities',
'depreciation',
'otherCashflowsFromInvestingActivities',
'dividendsPaid',
'changeToInventory',
'changeToAccountReceivables',
'salePurchaseOfStock',
'otherCashflowsFromFinancingActivities',
'changeToNetincome',
'capitalExpenditures',
'changeReceivables',
'cashFlowsOtherOperating',
'exchangeRateChanges',
'cashAndCashEquivalentsChanges',
'researchDevelopment',
'effectOfAccountingCharges',
'incomeBeforeTax',
'minorityInterest',
'sellingGeneralAdministrative',
'grossProfit',
'ebit',
'nonOperatingIncomeNetOther',
'operatingIncome',
'otherOperatingExpenses',
'interestExpense',
'taxProvision',
'interestIncome',
'netInterestIncome',
'extraordinaryItems',
'nonRecurring',
'otherItems',
'incomeTaxExpense',
'totalRevenue',
'totalOperatingExpenses',
'costOfRevenue',
'totalOtherIncomeExpenseNet',
'discontinuedOperations',
'netIncomeFromContinuingOps',
'netIncomeApplicableToCommonShares',
'preferredStockAndOtherAdjustments',
'num_shares'
]
print(len(cols))

stockData = pd.DataFrame(columns=cols)
count=0
for symbol in listOfStocks:
    count+=1
    print(count, ' Reading ', symbol, end='\r', flush=True)
    
    data = get_fundamental_data(symbol, myToken)
    

    if data == {}:
        continue
    if data['General']['Type'] == 'Common Stock':
        if data['outstandingShares']['quarterly'].keys():
            num_shares = np.mean([data['outstandingShares']['quarterly'][i]['shares'] for i in data['outstandingShares']['quarterly'].keys()])
        else:
            num_shares = np.nan
        if np.isnan(num_shares):
            num_shares = data['SharesStats']['SharesOutstanding']

        validDates = \
        data['Financials']['Income_Statement']['yearly'].keys() & \
        data['Financials']['Cash_Flow']['yearly'].keys() & \
        data['Financials']['Balance_Sheet']['yearly'].keys() #& \
        #data['Earnings']['Annual'].keys()

        for j in validDates:
            a = [data['Financials'][i]['yearly'][j] for i in data['Financials']] # BalanceSheet, CashFlow, IncomeStatement  
            b = {**data['General'], **data['SharesStats'], **a[0], **a[1], **a[2]} #, **data['Earnings']['Annual'][j]}
            b['num_shares'] = num_shares
            if 'Officers' in b:
                del b['Officers']
            if 'Description' in b:
                del b['Description']
            stockData = stockData.append(b, ignore_index=True)

stockData['date'] = pd.to_datetime(stockData['date'])

stockData[stockData['Code']=='IHRT']['date']

a=pd.DataFrame() # We will use 'a' as the name here as it keeps things less cluttered.
# we don't need variable 'a' for long anyway.
for tick in stockData['Code'].unique():
    a=a.append(stockData[stockData['Code']==tick].copy())

a

a['totalCurrentAssets']=a['totalCurrentAssets'].astype('float')
a['totalCurrentLiabilities']=a['totalCurrentLiabilities'].astype('float')
a['totalAssets']=a['totalAssets'].astype('float')
a['retainedEarnings']=a['retainedEarnings'].astype('float')

a['netIncome']=a['netIncome'].astype('float')
a['interestExpense']=a['interestExpense'].astype('float')
a['incomeTaxExpense']=a['incomeTaxExpense'].astype('float')
# StockData enhancement
a["EBIT"] = a["netIncome"] \
    - a["interestExpense"] \
    - a["incomeTaxExpense"]

a['EBIT']=a['EBIT'].astype('float')

a['totalStockholderEquity']=a['totalStockholderEquity'].astype('float')
a['totalLiab']=a['totalLiab'].astype('float')

a.fillna(0, inplace=True)

features = pd.DataFrame()

features['(CA-CL)/TA'] = ((a['totalCurrentAssets']\
                   - a['totalCurrentLiabilities'])\
                    /a['totalAssets']).clip(-4,4)
features['RE/TA'] = (a['retainedEarnings']/a['totalAssets']).clip(-20,5)
features['EBIT/TA'] = (a['EBIT']/a['totalAssets']).clip(-2,2)
features['BookEquity/TL'] = (a['totalStockholderEquity']/a['totalLiab']).clip(-10,100)

features['Ticker']=a['Code']
features['date']=a['date']

features

features[features['Ticker']=='BTU'].sort_values(by='date')

features[['(CA-CL)/TA', 'RE/TA', 'EBIT/TA', 'BookEquity/TL']]

# Matches the ticker and year bankrupt for values for 25 rows.
badStocks[['(CA-CL)/TA', 'RE/TA', 'EBIT/TA', 'BookEquity/TL']] = \
features[['(CA-CL)/TA', 'RE/TA', 'EBIT/TA', 'BookEquity/TL']]

badStocks

badStocks = pd.read_csv('badStocks.csv')
badStocks

"""
Get Data For Good Stocks From StockData File(Explained Later in Book)
"""

stockData=pd.read_csv('stockData.csv')
stockData=stockData[stockData['Exchange'].isin(['NYSE MKT', 'NASDAQ', 'NYSE ARCA', 'NYSE'])]
stockData['date'] = pd.to_datetime(stockData['date'])

# We will use 'a' as the name here as it keeps things less cluttered.
# we don't need variable 'a' for long anyway.
a = stockData[stockData['date'].between('01-01-2015','01-01-2021')].sample(25) # Grab 25 random companies
a

a['totalCurrentAssets']=a['totalCurrentAssets'].astype('float')
a['totalCurrentLiabilities']=a['totalCurrentLiabilities'].astype('float')
a['totalAssets']=a['totalAssets'].astype('float')
a['retainedEarnings']=a['retainedEarnings'].astype('float')

a['netIncome']=a['netIncome'].astype('float')
a['interestExpense']=a['interestExpense'].astype('float')
a['incomeTaxExpense']=a['incomeTaxExpense'].astype('float')
# StockData enhancement
a["EBIT"] = a["netIncome"] \
    - a["interestExpense"] \
    - a["incomeTaxExpense"]

a['EBIT']=a['EBIT'].astype('float')

a['totalStockholderEquity']=a['totalStockholderEquity'].astype('float')
a['totalLiab']=a['totalLiab'].astype('float')

a.fillna(0, inplace=True)

features = pd.DataFrame()

features['(CA-CL)/TA'] = ((a['totalCurrentAssets']\
                   - a['totalCurrentLiabilities'])\
                    /a['totalAssets']).clip(-4,4)
features['RE/TA'] = (a['retainedEarnings']/a['totalAssets']).clip(-20,5)
features['EBIT/TA'] = (a['EBIT']/a['totalAssets']).clip(-2,2)
features['BookEquity/TL'] = (a['totalStockholderEquity']/a['totalLiab']).clip(-10,100)

features['Ticker']=a['Code']
features['date']=a['date']

#features.to_csv('goodStocks.csv')

"""
See Both Good Stocks And Bad Stocks
"""

goodStocks = pd.read_csv('goodStocks.csv')
badStocks = pd.read_csv('badStocks.csv') # badStocks_backup.csv

goodStocks = goodStocks[['Ticker','date','(CA-CL)/TA','RE/TA','EBIT/TA','BookEquity/TL']]
goodStocks['Bankrupt']=0
goodStocks

#badStocks.drop(columns=['Ticker.1', 'Name', 'YearBankrupt'], inplace=True)
badStocks = badStocks[['Ticker','date','(CA-CL)/TA','RE/TA','EBIT/TA','BookEquity/TL']]
badStocks['Bankrupt'] = 1
badStocks

"""
Combining the Two in a Single DataFrame
"""
# To visualise some things easily.

stocksList = pd.concat([badStocks, goodStocks], ignore_index=True)
stocksList.drop(columns=['Ticker', 'date'], inplace=True)
bankruptList = stocksList['Bankrupt']
stocksList.drop(columns=['Bankrupt'], inplace=True)

stocksList

#stocksList.replace(0, goodStocks.mean())

plt.scatter(goodStocks['(CA-CL)/TA'], goodStocks['BookEquity/TL'])
plt.scatter(badStocks['(CA-CL)/TA'], badStocks['BookEquity/TL'], marker='X')
plt.xlabel('(CA-CL)/TA')
plt.ylabel('BookEquity/TL')
plt.xlim([-1,1])
plt.ylim([-1,1])
plt.grid()
plt.legend(['Non-Bankrupt Companies','Bankrupt Companies']);
plt.title('Bankrupt vs. Non-Bankrupt Companies\n With Two Features', fontsize=20);
