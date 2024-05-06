import numpy as np
import pandas
import sys
from datetime import datetime
#np.set_printoptions(threshold=sys.maxsize)

FILENAME = "new.csv"
DEBUG = True
VERBOSE = True

start = datetime.now()
#Obtains the dictionary of Symbols used in Smiles across the dataset, as well as setting them inside a dictionary as keys to be used for index lookup
def getSymbols(toxData):

    #get appropriate column
    smilesArray = toxData["smiles"]
    
    
    symbols = []
    #turn array into a single long string
    smilesString = ''.join(smilesArray)

    #for each character
    for index, char in enumerate(smilesString[:-1]):
        #skip c
        if char.islower() and char!='c':
            continue
        #if the next character is lowercase, not a c, this and next are alphanumerical and current one is uppercase
        if smilesString[index+1].islower() and smilesString[index+1]!='c' and smilesString[index+1].isalpha() and char.isalpha() and char.isupper():
            #create a combined symbol
            candidateSymbol = char + smilesString[index+1] 
        else:
            #in all other cases add the character directly
            candidateSymbol=char
        
        #if the symbol isn't in the list add it
        if candidateSymbol not in symbols:
            symbols.append(candidateSymbol)

    #add the last character if it's not lowercase (it was already added as partr of the previous) and not in symbols
    if smilesString[-1].isupper() and smilesString[-1] not in symbols:
        symbols.append(candidateSymbol)
    
    #generate indexes
    indexes = np.arange(len(symbols))
    #build dictionary and return it
    symbolsDict = dict(zip(symbols, indexes))
    return symbolsDict

#fills a tensor
def fillTensor(smile, inToxData):
    #get symbols from within Dataset
    containedSymbols = getSymbols(inToxData)
    #set the width of the tensor as the longest smile
    #not perfect, but good enough - counts all characters instead of symbols, but guarantees enough spaces
    inputTensorWidth = len(max(inToxData["smiles"], key=len))
    #set the height as the number of Symbols
    #perfect but not good enough
    inputtensorHeight = len(containedSymbols)
    
    #generate empty one hot matrix

    #zerosTensor = np.zeros((inputtensorHeight, inputTensorWidth))
    zerosTensor = np.zeros(inputtensorHeight * inputTensorWidth)
    #print("Zero Tensor Shape: ", zerosTensor.shape)
    smileArray = []

    #same as the getSymbols()
    for index, char in enumerate(smile[:-1]):
        if char.islower() and char!='c':
            continue
        if smile[index+1].islower() and smile[index+1]!='c' and smile[index+1].isalpha() and char.isalpha() and char.isupper():
            candidateSymbol = char + smile[index+1]
        else:
            candidateSymbol=char

        smileArray.append(candidateSymbol)
    #include last
    if smile[-1].isupper() or not smile[-1].isalpha() or smile[-1]=="c":
        smileArray.append(smile[-1])

    
    
    #index moves through the columns on each iteration
    #sets to 1 on any the index from the lookup symbol dictionary of the current column
    index=0
    for i in smileArray:
        
        #zerosTensor[containedSymbols[i]][index]=1
        zerosTensor[containedSymbols[i] * inputTensorWidth + index] = 1
        #print(i, "(", index, containedSymbols[i], ")", zerosTensor[containedSymbols[i]][index])
        index+=1
    #print(zerosTensor.sum())
    return zerosTensor


#Load data
if VERBOSE:
    print("Reading data from FILENAME ", FILENAME, ".")

#inToxData = pandas.read_csv(FILENAME)
inToxData = pandas.read_csv(FILENAME, encoding='latin1')
if VERBOSE:
    print("Data Loaded.")
    print("Enable DEBUG to show symbol list.")
#gets the list of symbols from input smiles

#print each dictionary entry in newline
if DEBUG:
    containedSymbols = getSymbols(inToxData)
    print("List of symbols and indices:")
    print("{" + "\n".join("{!r}: {!r},".format(v, k) for k, v in containedSymbols.items()) + "}")

if DEBUG:
    print("Test for Tensor Fill, should be 9: ", int(fillTensor("CC1=CC=C(C=C1)C", inToxData).sum().sum()))


#iterate over rows
# iterate over rows
dataset = []

if VERBOSE:
    print("Generating dataset, enable DEBUG to see each iteration output")
for index, row in inToxData.iterrows():
    # iterate over columns, skip the last one which stores the smiles
    datapoint = row.drop("smiles").values.tolist()  # Convert to a list of values

    # For smiles, transform into a one-hot encoded vector and append
    smile_one_hot = fillTensor(row["smiles"], inToxData)
    datapoint.append(smile_one_hot)  # Append the one-hot encoded vector to the data point

    # Append to the final output dataset, to be used as input for training
    dataset.append(datapoint)

if VERBOSE:
    print("Dataset converted in ", str(datetime.now() - start), "s.")

# Convert dataset to a NumPy array of one-dimensional arrays (one-hot encoded vectors)
dataset = np.array(dataset, dtype=object)


if VERBOSE:
    print(".NPZ saved.")
    print("Dataset converted in ", str(datetime.now()-start), "s.")
    import pandas as pd
    # Convert dataset to DataFrame for easier CSV saving
    np.set_printoptions(threshold=sys.maxsize)
    #dataset_df = pd.DataFrame((dataset))
    dataset_df = pd.DataFrame(np.squeeze(dataset))
    csv_filename = "stov.csv"
    dataset_df.to_csv(csv_filename, index=False)
if VERBOSE:
    print(".CSV saved.")
    txt_filename = "dataset.txt"
    np.set_printoptions(threshold=sys.maxsize)
    with open(txt_filename, "w",encoding="utf-8") as txt_file:
        for line in dataset:
            txt_file.write(str(line) + "\n")
    if VERBOSE:
        print(".TXT saved.")