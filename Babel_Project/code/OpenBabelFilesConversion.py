import string 
import os
# install pubchempy first and then import it to python as pcp
import pubchempy as pcp
#import openbabel, pybel
import sys
import pandas as pd
import subprocess
                
# conver mol file to smiles using opaenBable    
def convert_mol_to_smiles(molFile):
    #fileLocation = os.path.dirname(os.path.realpath(__file__))
    #smilesFilePath = os.path.join(".", "productsSmiles", molFile)
    
    smilesFilePath = molFile.replace('mol','smiles')
    f = open(smilesFilePath,'w')
    f.write('')
    f.close()
    # print molFilePath
    #print(smilesFilePath)
    
    os.system('babel -h -imol '+ molFile + ' -osmi ' + smilesFilePath + ' ---errorlevel 0')
    return smilesFilePath    
    

def read_File(fileAddress):
    # read the smiles file
    f = open(fileAddress,'r')
    product_smiles = f.read().rstrip()
    return product_smiles

# Use a SMILES string to search PubChem for mordetails
# such as Pubchem CID
def search_by_SMILES(smiles):
    # search the data base
    CIDs = []
    CIDDetails = []
    try:
        results = pcp.get_compounds(smiles, 'smiles')
        for result in results: 
            CIDs = result.cid
            CIDDetails = pcp.Compound.from_cid(result.cid).iupac_name
            break
    except:
        print ("serverError")
    
    # print CIDs
    # print CIDDetails
    if CIDs == None:
        details = ''
    else:
        details = str(CIDs) + ',' + str(CIDDetails)
    
    return details

# Use a PubChem CID to retreive a compound's chemical formula
def getCompoundFormula(cid):
    cidFormula = pcp.Compound.from_cid(cid).molecular_formula
    return cidFormula    
 
# Use a PubChem CID to retreive a compound's Inchi and InchKey
def getCompoundInchi(cid):
    cidInchi = pcp.Compound.from_cid(cid).inchi
    cidInchiKey = pcp.Compound.from_cid(cid).inchikey
    return cidInchi, cidInchiKey

# Use a PubChem CID to retreive a compound's SMILES
def getCompoundSmiles(cid):
    cidSmiles = pcp.Compound.from_cid(cid).canonical_smiles
    cidIsomericsmiles = pcp.Compound.from_cid(cid).isomeric_smiles
    return cidSmiles, cidIsomericsmiles

# Use a metabolite's name to retreive its PubChem ID
def getCompoundsPubChemID(metName):
    compound = pcp.get_compounds(metName, 'name')
    for cmp in compound:
        return str(cmp.cid)
            
            

# Use an InchiKey to retreive PubChem ID
def getPubchemIDFromInchiKey(inchikey):
    compound = pcp.get_compounds(inchikey, 'inchikey')    
    for cmp in compound:
        return cmp.cid
        
    
    
# Use a mole structure of a molecule to get its chemical formula
def convert_mol_to_formula(molFile):
    fileLocation = os.path.dirname(os.path.realpath(__file__))
    molFilePath = os.path.join(".", "productsMol", molFile)
    
    product = read_File(molFilePath)
    try:
        molObj = pybel.readstring("mol", product)
    except IOError:
        return ''
    return molObj.formula

    
# This function is to read a list of pubchem IDs from a file, get some details
# related to those IDs and save them to a file
def getPubChemIDsDetails():
    IDsFile = open('prodPubChemIDs.txt', 'r')
    pubChemIDsList = IDsFile.read().splitlines()
    
    writeFile = open('pubchemIDs_Details.txt', 'wb')
    writeFile.truncate()
    header = 'Pubchem ID; InChi; InChiKey; Canonical Smiles; Isomeric Smiles\n'
    writeFile.write(header)
    
    for pubchemID in pubChemIDsList:
        print (pubchemID)
        try:
            inchi, inchikey = getCompoundInchi(pubchemID)
            canonical_smiles, isomeric_smiles = getCompoundSmiles(pubchemID)
            line = pubchemID + ';' + inchi + ';' + inchikey + ';' + canonical_smiles + ';' + isomeric_smiles + '\n'
            writeFile.write(line)
        except:
            line = pubchemID + ';' + '-;-;-;-\n'
            writeFile.write(line)
    writeFile.close()
    
    
# This function is to read a list of pubchem IDs from a file, get some details
# related to those IDs and save them to a file
def getInchiKeyDetails():
    IDsFile = open('prodPubChemIDs.txt', 'r')
    inchikeyList = IDsFile.read().splitlines()
    
    writeFile = open('inchikey_Details.csv', 'wb')
    writeFile.truncate()
    header = 'Pubchem ID, InChiKey\n'
    writeFile.write(header)
    
    count = 0
    for inchikey in inchikeyList:
        count += 1 
        print (count )
        pubchemID = getPubchemIDFromInchiKey(inchikey)
        line = str(pubchemID) + ',' + inchikey + '\n'
        writeFile.write(line)

    writeFile.close()    
    
def get_unique_entries():
    writeFile = open('prodPubChemIDs_details_unique.csv', 'wb')
    writeFile.truncate()
    
    with open('prodPubChemIDs_details.csv', 'r') as file:
        file_lines = file.read().splitlines()
        
        writeFile.write(file_lines[0]+'\n')
        
        file_lines.pop()
        
        unique_entries = []
        unique_entries.append(file_lines[0])
        for line in file_lines:
            if line not in unique_entries:
                unique_entries.append(line)
                writeFile.write(line+'\n')
    writeFile.close()
    
def main_function(molFile):
    #print (molFile)
    #fileLocation = os.path.dirname(os.path.realpath(__file__))
    #molFilePath = os.path.join(".", "productsMol", molFile)
    #print(fileLocation)
    #print(molFilePath)
    matches = {}
    path_to_smilesFile = convert_mol_to_smiles(molFile)
    #print(path_to_smilesFile)
    product_smiles = read_File(path_to_smilesFile)
    CIDDetails = search_by_SMILES(product_smiles)
    
    return CIDDetails


def convertMoltoPubChemID(molFile, smileFile,pubChemIDFile):
    os.system('babel -h -imol '+ molFile + ' -osmi ' + smileFile + ' ---errorlevel 0')


    # p = subprocess.Popen('babel -h -imol '+ molFile + ' -osmi ' + smileFile + ' ---errorlevel 0')
   # p.wait()
    product_smiles = read_File(smileFile)
    CIDDetails = search_by_SMILES(product_smiles)
    with open(pubChemIDFile, 'w') as f:
        print(CIDDetails, file=f)




if __name__ == '__main__':
    # Map command line arguments to function arguments.
    #convertMoltoPubChemID(*sys.argv[1:])
#
#     molFile = '_C00257_1.1.1.35/product_1.mol'
#     smileFile = '_C00257_1.1.1.35/product_1.smile'
#     pubchemIDFile = '_C00257_1.1.1.35/product_1.pubchemID'
#     convertMoltoPubChemID(molFile, smileFile, pubchemIDFile)
    molFileBase = "/home/sunny/Desktop/Babel_Project" ## Change it to your project path. Please don't keep SPACE in folder name hence path.
    molFilePhase1 = os.path.join(molFileBase, 'Phase1')
    molFilePhase2 = os.path.join(molFileBase, 'Phase2')
    report = []
    for folder in os.listdir(molFilePhase1):
        if '.DS_Store' not in os.path.join(molFilePhase2,folder):
            mol_files = os.listdir(os.path.join(molFilePhase1,folder))
            for mol_file in mol_files:
                if '.mol' in mol_file:
                    full_mol_file_path = os.path.join(molFilePhase1, folder, mol_file)
                    full_cid_file_path = full_mol_file_path.replace('mol', 'pubchemID')
                    CIDDetails = main_function(full_mol_file_path)
                    f = open(full_cid_file_path, 'w')
                    f.write(str(CIDDetails))
                    if CIDDetails and CIDDetails.split(',')[0].isdigit():
                        cid = CIDDetails.split(',')[0]
                        print(cid)
                        cid_formula = getCompoundFormula(cid)
                        cidInchi, cidInchiKey = getCompoundInchi(cid)
                        cidSmiles, cidIsomericsmiles = getCompoundSmiles(cid)
                        url  = 'https://pubchem.ncbi.nlm.nih.gov/compound/'+str(cid)
                    else:
                        cid,cid_formula,cidInchi, cidInchiKey,cidSmiles, cidIsomericsmiles,url = '','','','','','',''
                    report.append([folder,folder+'_'+mol_file,cid,cid_formula,cidInchi,cidInchiKey,cidSmiles,cidIsomericsmiles,url])
                    
                    f.close()
    
    df = pd.DataFrame(report,columns = ['FolderName','MolFileName','CompoundId','CompundFormula','Inchi','InchiKey','CIDSmiles','Isomericsmiles','URL'])
    df.to_csv('Phase1.csv',index=False)
    
    report = []
    for folder in os.listdir(molFilePhase2):
        if '.DS_Store' not in os.path.join(molFilePhase2,folder):
            mol_files = os.listdir(os.path.join(molFilePhase2,folder))
            for mol_file in mol_files:
                if '.mol' in mol_file:
                    full_mol_file_path = os.path.join(molFilePhase2, folder, mol_file)
                    full_cid_file_path = full_mol_file_path.replace('mol', 'pubchemID')
                    CIDDetails = main_function(full_mol_file_path)
                    f = open(full_cid_file_path, 'w')
                    f.write(str(CIDDetails))
                    if CIDDetails and CIDDetails.split(',')[0].isdigit():
                        cid = CIDDetails.split(',')[0]
                        cid_formula = getCompoundFormula(cid)
                        cidInchi, cidInchiKey = getCompoundInchi(cid)
                        cidSmiles, cidIsomericsmiles = getCompoundSmiles(cid)
                        url  = 'https://pubchem.ncbi.nlm.nih.gov/compound/'+str(cid)
                    else:
                        cid,cid_formula,cidInchi, cidInchiKey,cidSmiles, cidIsomericsmiles,url = '','','','','','',''
                    report.append([folder,folder+'_'+mol_file,cid,cid_formula,cidInchi,cidInchiKey,cidSmiles,cidIsomericsmiles,url])
                    
                    f.close()
    df = pd.DataFrame(report,columns = ['FolderName','MolFileName','CompoundId','CompundFormula','Inchi','InchiKey','CIDSmiles','Isomericsmiles','URL'])
    df.to_csv('Phase2.csv',index=False)
