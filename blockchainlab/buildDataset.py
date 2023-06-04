from web3 import Web3

# Set up a connection to the blockchain
ganache_url = "http://127.0.0.1:7545"
web3 = Web3(Web3.HTTPProvider(ganache_url))

# Open the participantDetails.csv file to read participant details
fileHandleIN=open("participantDetails.csv","r")

# Open the dataSet.csv file to store our constructed dataset
fileHandleOUT=open("dataSet.csv","w")