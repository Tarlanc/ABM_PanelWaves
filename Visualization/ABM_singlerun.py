# -*- coding: iso-8859-1 -*-

import time
import math
import random
import os
try: ##if Python Version 3.x
    from tkinter import *
    from tkinter import messagebox
    from tkinter import filedialog
    py_version = 3
except: ##If Python Version 2.7
    from Tkinter import *
    import tkMessageBox
    import tkFileDialog
    py_version = 2
    
random.seed()

global verb_level
verb_level = 3 ## Level of statements to be written to the console. Higher levels mean more output

####################################################
###
### Basic functions for data input, output, and visualization
###
####################################################


def niceprint_tree(tdic, einr = 0, inherit = '', spc = '    ', trunc=0, lists=True):
    ## This function takes a variable in dictionary or list format and transforms the values
    ## to a readable format in one single string variable.
    if type(tdic)==dict:
        if inherit == '': inherit = '{\n'
        for k in sorted(tdic.keys()):
            if type(tdic[k]) == dict:
                inherit = inherit + einr*spc + str(k) + ': ' + niceprint_tree(tdic[k],einr+1,inherit = '{\n',trunc=trunc)
            elif type(tdic[k]) == list and lists:
                inherit = inherit + einr*spc + str(k) + ': ' + niceprint_tree(tdic[k],einr+1,inherit = '[\n',trunc=trunc)
            else:
                value = tdic[k]
                if type(value)==str:
                    value = "'"+value+"'"
                elif type(value) in [int,float]:
                    value = str(value)
                elif type(value) == list:
                    value = str(value)
                else:
                    value = str(value)
                if len(value) > trunc and trunc > 0:
                    tail = int(trunc/2)
                    value = value[:tail] + '...'+value[-tail:] + ' ('+str(len(value))+' characters)'                  
                inherit = inherit + einr*spc + str(k) + ': '+ value + '\n'
        inherit = inherit + einr*spc + '}\n'
    elif type(tdic)==list:
        if inherit == '': inherit = '[\n'
        for e in tdic:
            if type(e) == dict:
                inherit = inherit + einr*spc + niceprint_tree(e,einr+1,inherit = spc+'{\n',trunc=trunc)
            elif type(e) == list and lists:
                inherit = inherit + einr*spc + niceprint_tree(e,einr+1,inherit = spc+'[\n',trunc=trunc)
            else:
                value = e
                if type(value)==str:
                    value = "'"+value+"'"
                elif type(value) in [int,float]:
                    value = str(value)
                elif type(value) == list:
                    value = str(value)
                else:
                    value = str(value)
                if len(value) > trunc and trunc > 0:
                    tail = int(trunc/2)
                    value = value[:tail] + '...'+value[-tail:] + ' ('+str(len(value))+' characters)'                 
                inherit = inherit + einr*spc + value + ',\n'
        inherit = inherit + einr*spc + ']\n'   
    return inherit

def get_data(fname):
    ## Read a table from a tab-spaced text document
    infile = open(fname,'r')
    zeilen = infile.readlines()
    infile.close()
    header = zeilen[0][:-1]
    daten = {}
    variablen = header.split('\t')
    for v in variablen:
        daten[v] = []
    for z in zeilen[1:]:
        datenzeile = z[:-1].split('\t')
        if len(datenzeile) == len(variablen):
            for i in range(0,len(variablen)):
                daten[variablen[i]].append(datenzeile[i])
        else:
            print('Ungültige Zeile: "'+z+'"')
    return [daten,variablen]

def write_data(daten,fname):
    ## Write a table to a tab-spaced text document
    outfile = open(fname,'w')
    variablen = daten[1]
    ddic = daten[0]
    header = '\t'.join(variablen)
    outfile.write(header+'\n')
    for zeile in range(0,len(ddic[variablen[0]])):
        datenzeile = []
        for v in variablen:
            datenzeile.append(str(ddic[v][zeile]))
        outfile.write('\t'.join(datenzeile)+'\n')
    outfile.close()

def write_sample(infile,outfile,size=100):
    ## Draw a sample with repetition from a data file and store the result
    source = open(infile,'r')
    inlines = source.readlines()
    source.close()
    target = open(outfile,'w')
    target.write(inlines[0])
    maxline = len(inlines)-1
    for b in range(size):
        sel = random.randint(1,maxline)
        target.write(inlines[sel])
    target.close()


def write_samples(infile,train_outfile,test_outfile,size=100):
    ## Draw a sample with repetition from a data file and store the result
    source = open(infile,'r')
    inlines = source.readlines()
    source.close()
    target1 = open(train_outfile,'w') ## Open Training Data
    target1.write(inlines[0])
    target2 = open(test_outfile,'w')  ## Open Test Data
    target2.write(inlines[0])

    cases = range(1,len(inlines))

    testcases = random.sample(cases,size)

    for i in cases:
        if i in testcases:
            target1.write(inlines[i])
        else:
            target2.write(inlines[i])
            
    target1.close()
    target2.close()
    

def get_unique(liste):
    ## Return unique values from a list
    td = {}
    for element in liste:
        td[element] = 0
    return sorted(td.keys())

def verb(zeile,nl=1,verbose=0):
    ## Print statements
    global verb_level
    if verb_level >= verbose:
        if nl == 0:
            print(zeile),
        else:
            print(zeile)


def c_mittel(liste,ln=0):
    ##Calculate the mean value of a numeric list
    anz = 0
    summe = 0.0
    for e in liste:
        try:
            if ln == 1:
                summe = summe + math.log(float(e))
            else:
                summe = summe + float(e)
            anz = anz + 1
        except:
            summe = summe

    if anz > 0:
        mittel = summe/anz
    else:
        mittel = 0
    return mittel

def c_sdev(liste,ln=0):
    #Calculate the standard deviation of a numeric list
    m = c_mittel(liste,ln)
    summe = 0.0
    anz = 0
    if not m == 0:
        for e in liste:
            try:
                if ln == 1:
                    summe = summe + (math.log(e)-m)**2
                else:
                    summe = summe + (e-m)**2
                anz = anz + 1
            except:
                summe = summe

        if anz > 1:
            sd = (summe/(anz-1))**.5
        else:
            sd = 0
    else:
        sd = 0

    return sd

def c_low(liste,alpha=0.05):
    #Calculate the lower bound of the 1-alpha CI
    tail = int(len(liste)*alpha)
    vlist = []
    for v in liste:
        try:
            vlist.append(float(v))
        except:
            a = 1
    outval = sorted(vlist)[tail]
    return outval

def c_hig(liste,alpha=0.05):
    #Calculate the lower bound of the 1-alpha CI
    tail = int(len(liste)*alpha)
    if tail == 0: tail = 1
    vlist = []
    for v in liste:
        try:
            vlist.append(float(v))
        except:
            a = 1
    outval = sorted(vlist)[-tail]
    return outval


####################################################
###
### Simulation
###
####################################################

## Basic functions to be used in the simulation

def vectorize(pdic,var):
    ## Return a vector of values from all agents
    outlist = []
    for p in sorted(pdic.keys()):
        outlist.append(pdic[p][var])
    return outlist

def calc_pearson(pdic,v1,v2):
    ## Compute the pearson correlation between two variables for all agents
    sv1 = 0.0
    sv2 = 0.0
    sdv1 = 0.0
    sdv2 = 0.0
    cov = 0.0
    anz = 0
    for p in pdic.keys():
        if type(pdic[p][v1]) == float and type(pdic[p][v2]) == float:
            sv1 = sv1 + pdic[p][v1]
            sv2 = sv2 + pdic[p][v2]
            anz = anz + 1
    mv1 = sv1 / anz
    mv2 = sv2 / anz
    for p in pdic.keys():
        if type(pdic[p][v1]) == float and type(pdic[p][v2]) == float:
            sdv1 = sdv1 + (pdic[p][v1]-mv1)**2
            sdv2 = sdv2 + (pdic[p][v2]-mv2)**2
            cov = cov + (pdic[p][v1]-mv1)*(pdic[p][v2]-mv2)
    sdv1 = (sdv1/anz)**.5
    sdv2 = (sdv2/anz)**.5
    cov = (cov/anz)
    if sdv1*sdv2 > 0:
        pcorr = cov/(sdv1*sdv2)
    else:
        pcorr = '-'

    return pcorr

def calc_identity(pdic,v1,v2):
    ## Compute the inverted mean square deviation for two variables for all agents.
    sqsum = 0.0
    anz = 0
    for p in pdic.keys():
        val1 = pdic[p][v1]
        val2 = pdic[p][v2]
        if type(val1) == float and type(val2) == float:
            sqsum = sqsum + (val1-val2)**2
            anz = anz + 1
    if anz > 0:
        sqmean = sqsum/anz
    else:
        sqmean = 1000

    return 1.0/sqmean


## Initialization of the model

def initialize(pfile,mfile,verbose=1):
    ## Load the data from tables and pass it to initialize agents (pdic) and media environment (mdic)
    if verbose == 1:
        verb('Loading all Data...')
    pdata = get_data(pfile)[0]
    mdata = get_data(mfile)[0]

    if verbose ==1:
        verb('Variables in Individual Data: '+str(sorted(pdata.keys())))
        verb('Veriables in Media Data: '+str(sorted(mdata.keys())))
        verb('\nCreating dictionaries')
    mdic = create_mdic(mdata)
    pdic = create_pdic(pdata,mdic.keys(),verbose=verbose)
    if verbose == 1:
        verb('Media identified: '+str(mdic.keys()))
        verb(str(len(pdic.keys()))+' Cases identified')
    return (pdic,mdic)


def create_mdic(data):
    ## Initiate the media dictionary which serves as environment
    timerange = get_unique(data['Week_Excel'])
    for i in range(len(timerange)):
        timerange[i] = int(timerange[i])
    mindate = int(timerange[0])
    maxdate = int(timerange[-1])
    arguments = ['Arg_Dicho_1','Arg_Dicho_2','Arg_Dicho_9']
    mdic = {}
    for m in get_unique(data['Medium']):
        mdic[m] = {}
        for d in timerange:
            mdic[m][d] = {}
            for a in arguments:
                mdic[m][d][a] = 0
            mdic[m][d]['Bias']= 0

    for i in range(len(data['Medium'])):
        try:
            d = int(data['Week_Excel'][i])
        except:
            d = maxdate
        m = data['Medium'][i]       
        for arg in arguments:
            try:
                a = float(data[arg][i])
            except:
                a = 0
            mdic[m][d][arg]=a
        mdic[m][d]['Bias']=(mdic[m][d]['Arg_Dicho_1']-mdic[m][d]['Arg_Dicho_2'])/(mdic[m][d]['Arg_Dicho_1']+mdic[m][d]['Arg_Dicho_2']+mdic[m][d]['Arg_Dicho_9'])

    #verb(niceprint_tree(mdic))
    return mdic


def create_pdic(data,media,verbose = 1):
    ##Create the population dictionary which holds all agents
    invalid_cases = 0
    valid_vases = 0
    persvar = ['intnum','sex','age','Zeitung_W1','TV_W1','Zeitung_W2','TV_W2','Zeitung_W3','TV_W3',
               'Einstellung_W1','Einstellung_W2','Einstellung_W3','Einstellung_Pro1','Einstellung_Pro2','Einstellung_Pro3',
               'Cert1','Cert2','Cert3','Media_Reliance','Disk_Reliance','dim_bildung','dim_x','dim_y','dim_lr','dim_spr']
    medvar = ['TV_W1', 'TV_W2', 'TV_W3', 'Zeitung_W1',
              'Zeitung_W2', 'Zeitung_W3']
   
    pdic = {}
    for i in range(len(data['intnum'])):
        valid = 1
        ident = data['intnum'][i]
        try:
            v = float(data['Einstellung_Pro1'][i]) ##Remove agents of whom we do not know the attitude
        except:
            valid = 0
        if valid == 1:
            valid_cases = valid_vases + 1
            pdic[ident] = {}
            for p in persvar:
                try:
                    pdic[ident][p] = float(data[p][i])
                except:
                    pdic[ident][p] = 'NA'
               
            pdic[ident]['Einst'] = pdic[ident]['Einstellung_Pro1'] #Initalize the dynamic attitude
            if type(pdic[ident]['Einst']) == str:
                pdic[ident]['Einst'] = 0.0
            pdic[ident]['Media'] = []
            for m in medvar:
                if data[m][i] in media:
                    pdic[ident]['Media'].append(data[m][i])
            pdic[ident]['Media'] = get_unique(pdic[ident]['Media'])
        else:
            invalid_cases = invalid_cases + 1
    if verbose == 1:
        verb(str(invalid_cases)+' Invalid cases identified')
        verb(str(valid_cases)+' Agents in Simulation')
 
    return pdic


## Computation of impacts

def calc_simpact(pdic,p1,param):
    ## Compute the social impact on one agent (p1)
    pl = list(pdic.keys())
    pl.remove(p1)
    bias = 0.0
    anz = 1.0
    
    for p2 in pl:
        if pdic[p1]['dim_spr'] == pdic[p2]['dim_spr']:
            dist_ort = (pdic[p1]['dim_x']-pdic[p2]['dim_x'])**2 + (pdic[p1]['dim_y']-pdic[p2]['dim_y'])**2
            try:
                dist_lr = (pdic[p1]['dim_lr']-pdic[p2]['dim_lr'])**2
            except:
                dist_lr = 0.5
            distance = dist_ort*param['coeff_d']**2+dist_lr
            if distance < 1:
                distance = 1
            
            bias = bias + (pdic[p2]['Einst']-pdic[p1]['Einst'])/distance
            anz = anz + 1/distance
     
    try:
        impact = bias / anz
    except:
        impact = 0.0
        verb('ERROR: Div by 0 in SI_impact (imp_pers, anz_pers, imp_supp, anz_supp): '+str(imp_pers)+'; '+str(anz_pers)+'; '+str(imp_supp)+'; '+str(anz_supp))   
    return impact

def calc_mimpact(mdic,medien,tag,param):
    ## Compute the media impact, using a list of media (medien)
    mimpact = 0.0
    anz = 0
    for m in medien:
        mimpact = mimpact + mdic[m][tag]['Bias']
        anz = anz + 1
    if anz > 1:
        mimpact = mimpact / anz
    return mimpact


def simulate(pdic, mdic, steps, param):
    ## Actual simulation, using the population, media environment, points in time, and parameters
    results = []
    ts = time.time()
    outsim = {}
    outsimvar=['Initial']
    outsim['Initial']=vectorize(pdic,'Einst')
    b_cert = param['coeff_e'] ## Use the coefficients directly
    b_reli = param['coeff_a']
    careers = {}

    career_ind = ["31101","21738","62230",
                  "46538","13279","24551",
                  "51926","78122","24569",
                  "50975","60461","90809"] ## Interview number of 10 different agents

    for c in career_ind:
        careers[c] = [pdic[c]['Einst']]
    

    for d in steps:
        for p in pdic.keys():
            MI = calc_mimpact(mdic,pdic[p]['Media'],d,param)
            SI = calc_simpact(pdic,p,param)
            try:
                Msusc = param['coeff_b']+b_cert*pdic[p]['Cert1']+b_reli*pdic[p]['Media_Reliance']
            except:
                Msusc = 0.0

            try:
                Ssusc = param['coeff_c']+b_cert*pdic[p]['Cert1']
            except:
                Ssusc = 0.0
                
            pdic[p]['dEinst']=MI*Msusc + SI*Ssusc

            #print(pdic[p]['dEinst'])

        for p in pdic.keys():
            pdic[p]['Einst'] = pdic[p]['Einst']+pdic[p]['dEinst']
            if pdic[p]['Einst']>2: pdic[p]['Einst']=2
            if pdic[p]['Einst']<-2: pdic[p]['Einst']=-2

            if p in career_ind:
                careers[p].append(pdic[p]['Einst'])
            
        r = calc_identity(pdic,'Einst','Einstellung_Pro2')
        verb('Mean square sum on day: '+str(d)+' = '+str(1.0/r),verbose=3)

        verb('.',nl=0,verbose=2)
        results.append(r)
        outsim[str(d)]=vectorize(pdic,'Einst')
        outsimvar.append(str(d))
        
    pearson_r = calc_pearson(pdic,'Einst','Einstellung_Pro2')

    #write_data(outsim,outsimvar,'tmp_outsim.txt') ## Write a temporary result of the simulation

    ts = time.time()-ts
    verb(' Simulation finished in: '+str(ts)+' Seconds',verbose=1)
    tr = open('tmp_w1_bs.txt','a')
    for p in sorted(param.keys()):
        tr.write(str(param[p])+'\t')
    for r in results:
        tr.write(str(r)+'\t')
    tr.write('\n')
    tr.close()
    
    make_visual(pdic)
    
    return [results[-1],pearson_r,careers]

def make_visual(pdic):
    global verb_level
    if verb_level > 1:
        swissgrid = {}
        for x in range(47,83):
            swissgrid[x*10] = {}
            for y in range(8,30):
                swissgrid[x*10][y*10] = {}
                swissgrid[x*10][y*10]['Values']=[]
                swissgrid[x*10][y*10]['M']=0
                swissgrid[x*10][y*10]['Col']='#ffffff'

        for p in pdic:
            x = int(pdic[p]['dim_x']*10)*10
            y = int(pdic[p]['dim_y']*10)*10
            swissgrid[x][y]['Values'].append(pdic[p]['Einst'])
            swissgrid[x][y]['Col']='#ffaaaa'
       
        master = Tk()
        w = Canvas(master,bg="#ffffff",width=850,height=600)
        w.pack()
        for x in swissgrid.keys():
            for y in swissgrid[x].keys():
                if len(swissgrid[x][y]['Values']) > 0:
                    val = c_mittel(swissgrid[x][y]['Values'])/2 ## Value between -1 (contra) and +1 (pro)

                    maincol = abs(val)*.3+.7
                    offcol = (1-abs(val))*.7

                    #print(val,maincol,offcol)

                    if val > 0:
                        r = 255*offcol
                        g = 255*maincol
                        b = 255*offcol
                    else:
                        r = 255*maincol
                        g = 255*offcol
                        b = 255*offcol

                    rc = hex(int(r))[2:]
                    gc = hex(int(g))[2:]
                    bc = hex(int(b))[2:]

                    if len(rc)==1: rc = "0"+rc
                    if len(gc)==1: gc = "0"+gc
                    if len(bc)==1: bc = "0"+bc

                    swissgrid[x][y]['Col'] = '#'+rc+gc+bc

                    #print(val,r,g,b,rc,gc,bc)
                    
##                    r = 255*(1-(val/2+.5)**2)
##                    g = 255*(1-(val/2-.5)**2)
##                    if r < 0: r = 0
##                    if g < 0: g = 0
##
##                    rot = hex(int(r))[2:]
##                    gru = hex(int(g))[2:]
##                    if len(rot)==1: rot = '0'+rot
##                    if len(gru)==1: gru = '0'+gru
##
##                    swissgrid[x][y]['Col'] = '#'+rot+gru+'00'
                else:
                    swissgrid[x][y]['Col'] = '#ffffff'
                    
                w.create_rectangle(2*x-900, 700-2*y, 2*x-880, 680-2*y, fill=swissgrid[x][y]['Col'])
        w.update()
        mainloop()
        



####################################################
###
### Main program
###
####################################################


simdates=[38927,38934,38941,38948,38955,38962] #Dates for Phase 1
var_e1 = 'Einstellung_Pro1'
var_e2 = 'Einstellung_Pro2'

dics = initialize('Survey_Data.dat','Content_Data.dat',verbose=0) ## Initialize the simulation
pcorr_init_r = calc_pearson(dics[0],var_e1,var_e2) ## Get the initial correlation of attitudes
pcorr_init_id = calc_identity(dics[0],var_e1,var_e2) ## Get the initial agreement of attitudes

verb('\nInitial inverted squaresum: '+str(pcorr_init_id),verbose=1)
verb('\nInitial Correlation of Attitudes: '+str(pcorr_init_r),verbose=1)

## Optimal values
param = {'coeff_a':.1725, 'coeff_b':-0.0903, 'coeff_c':0.0705,
         'coeff_d':2.189, 'coeff_e':-.043}

## No social and media influence
##param = {'coeff_a':0.0, 'coeff_b':0.0, 'coeff_c':0.0,
##         'coeff_d':1.0, 'coeff_e':0.0}

## Only social influence
##param = {'coeff_a':0.0, 'coeff_b':0.0, 'coeff_c':0.1,
##         'coeff_d':1.0, 'coeff_e':0.0}

## Only media influence
##param = {'coeff_a':0.0, 'coeff_b':0.8, 'coeff_c':0.0,
##         'coeff_d':1.0, 'coeff_e':0.0}

## Full social and media influence
##param = {'coeff_a':0.0, 'coeff_b':0.8, 'coeff_c':0.1,
##         'coeff_d':1.0, 'coeff_e':0.0}


verb('\nParameters: '+str(param),verbose=1)
simresult = simulate(dics[0],dics[1],simdates,param) ## Call the simulation and return the final agreement

inex = 1/pcorr_init_id ## Mean square deviation between the waves
exp = 1/simresult[0] ## Mean square deviation after simulation
dr2 = (inex-exp)/inex*100 ## Proportional reduction of error: Percent of explained variance by simulation

verb('delta R2 (explained variance by simulation): '+"{0:2.2f}%".format(dr2),verbose=1)
            
#print(niceprint_tree(simresult[2]))

write_data([simresult[2],sorted(simresult[2].keys())],"Careers.txt")
