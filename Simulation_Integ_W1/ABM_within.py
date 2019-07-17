# -*- coding: iso-8859-1 -*-

import time
import math
import random
import os
random.seed()

global verb_level
verb_level = 0 ## Level of statements to be written to the console. Higher levels mean more output

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
            datenzeile.append(ddic[v][zeile])
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
### Evolutionary algorithm
###
####################################################


def init_settings(fname,manualset,force=0):
    ## Initialize the basic settings for the evolutionary algorithm.
    ## The settings are written to a file (fname). If the file
    ## already exists and force is set to 1, the settings from
    ## the file are taken instead of the manually set values.
    ## This function may be used to change the boundary conditions
    ## of the evolutionary algorithm in mid-execution.
    
    global g_settings
    if force == 0:
        try:
            gset = open(fname,'r')
            gs = gset.readline()
            gset.close()
            while not gs[-1]=='}':
                gs = gs[:-1]
            gs = eval(gs)
            print('Settings loaded: '+str([gs]))
        except:
            print('There are no previously stored settings. Setting current.')
            gset = open(fname,'w')
            gset.write(str(manualset))
            gset.write('\n')
            gset.close()
            gs = manualset
    else:
            gset = open(fname,'w')
            gset.write(str(manualset))
            gset.write('\n')
            gset.close()
            gs = manualset      
    g_settings = gs

def evol_manhattan(prior,mutation=0):
    ## Use the prior distributions (m and sd) to compute a new value
    ## If mutation is set to 1, the standard deviation is doubled, allowing
    ## for parameters outside the usual range.
    
    v = prior['Vert']
    mut = random.random()
    m = prior['M']
    if mut < mutation: ## In case of mutation events, increase the variance
        sd = prior['SD']*2
    else:
        sd = prior['SD']
    
    if v == 'norm':
        a = random.normalvariate(m,sd)
    elif v == 'normp':
        a = random.normalvariate(m,sd)
        if a < 0: a = 0 ## Force positive numbers. 
    elif v == 'enorm':
        a = math.exp(random.normalvariate(m,sd))
    elif v == 'const':
        a = m
    return a

def evol_initialize(ps,anz=100):
    ## Initialize the prior dictionary for the evolutionary algorithm
    ## Each parameter is a key in this dictionary and holds previous
    ## and current values, as well as distribution information
    parlist = ps.keys()
    prior = {}
    for p in parlist:
        prior[p] = {}
        prior[p]['M'] = ps[p][0]
        prior[p]['SD'] = ps[p][1]
        prior[p]['Vert']=ps[p][2]
        prior[p]['Prev']=[]
        prior[p]['Curr']=[]
        prior[p]['Score']=0
        for i in range(anz):
            prior[p]['Curr'].append(evol_manhattan(prior[p]))
    return prior

def evol_write_result(prior,fname='results.txt'):
    ## Write the result of one generation of parameter sets to an external file.
    try:
        u1 = open(fname,'r')
        u1.close()
    except:
        u1 = open(fname,'w')
        for p in sorted(prior.keys()):
            if not p == 'Results':
                u1.write(p+'\t')
        u1.write('Result\n')
        u1.close()
    outfile = open(fname,'a')
    for i in range(len(prior['Results'])):
        for p in sorted(prior.keys()):
            if not p == 'Results':
                outfile.write(str(prior[p]['Curr'][i])+'\t')
        outfile.write(str(prior['Results'][i])+'\n')
    outfile.close()
            

def evol_update(prior,special_ind,gen_size=20,length=100,mutation=0.1,fname='upd.txt'):
    ## Eliminate parameter sets with low results an generate new sets.
    ## At the end, an updated version of the prior dictionary is returned.
    ## A detailed report for each parameter set is written to an external file for inspection
    global g_settings
    try:
        u1 = open(fname,'r')
        u1.close()
    except:
        u1 = open(fname,'w')
        u1.write('Timestamp\tM_Result\tSD_Result\tLow_Result\tHig_Result\t')
        for p in sorted(prior.keys()):
            if not p == 'Results':
                u1.write('M_'+p+'\tSD_'+p+'\tLow_'+p+'\tHig_'+p+'\t')
        u1.write('\n')
        u1.close()
    
    upd = open(fname,'a')
    while -1000 in prior['Results']:
        prior['Results'].remove(-1000) ###Remove failed simulations
    rm = c_mittel(prior['Results'])
    rs = c_sdev(prior['Results'])
    rl = c_low(prior['Results'])
    rh = c_hig(prior['Results'])
    upd.write(time.ctime()+'\t')
    upd.write(str(rm)+'\t'+str(rs)+'\t'+str(rl)+'\t'+str(rh)+'\t') ##Write the result of the simulation before cleaning up the gene pool
    del prior['Results']  ##Remove results from the prior distribution for next run.
    for p in sorted(prior.keys()):
        keepers = []
        for i in range(len(prior[p]['Curr'])):
            if not i in special_ind[0]:
                prior[p]['Prev'].append(prior[p]['Curr'][i]) ##Kill losers, retain others as previous candidates
                if i in special_ind[1]:
                    keepers.append(prior[p]['Curr'][i])
                
        if len(prior[p]['Prev'])>length:
            prior[p]['Prev'] = prior[p]['Prev'][-length:]
        if prior[p]['Vert'] in ['norm','pnorm','normp']:
            prior[p]['M']=c_mittel(prior[p]['Prev'])
            prior[p]['SD']=c_sdev(prior[p]['Prev'])
            prior[p]['Low']=c_low(prior[p]['Prev'])
            prior[p]['Hig']=c_hig(prior[p]['Prev'])
        elif prior[p]['Vert'] in ['const']:
            prior[p]['M']=prior[p]['M']
            prior[p]['SD']=0
            prior[p]['Low']=c_low(prior[p]['Prev'])
            prior[p]['Hig']=c_hig(prior[p]['Prev'])            
        else:
            prior[p]['M']=c_mittel(prior[p]['Prev'],ln=1)
            prior[p]['SD']=c_sdev(prior[p]['Prev'],ln=1)            
            prior[p]['Low']=c_low(prior[p]['Prev'])
            prior[p]['Hig']=c_hig(prior[p]['Prev'])

        if 'Overwrite' in g_settings.keys():
            ## If the settings include an Overwrite-dictionary, the means and standard deviations
            ## for specified parameters will be overwritten. Using this hack, parameters may be
            ## Changed in mid-execution. You may treat this option as a "God of Evolution" protocol.
            if p in g_settings['Overwrite'].keys():
                print('Overwriting parameter "'+p+'". Old settings: M='+str(prior[p]['M'])+'; SD='+str(prior[p]['SD']))
                if 'M' in g_settings['Overwrite'][p].keys(): prior[p]['M'] = g_settings['Overwrite'][p]['M']
                if 'SD' in g_settings['Overwrite'][p].keys(): prior[p]['SD'] = g_settings['Overwrite'][p]['SD']
                print(' --> New settings: M='+str(prior[p]['M'])+'; SD='+str(prior[p]['SD']))

        prior[p]['Curr'] = keepers
        for i in range(len(keepers),gen_size):
            prior[p]['Curr'].append(evol_manhattan(prior[p],mutation))        
        upd.write(str(prior[p]['M'])+'\t'+str(prior[p]['SD'])+'\t'+str(prior[p]['Low'])+'\t'+str(prior[p]['Hig'])+'\t')
    upd.write('\n')
    upd.close()
    return prior

def evol_fail(rlist,share,legshare=0.0):
    ## Identify the failing candidates and return their IDs
    anz = int(len(rlist)*share)
    anzwin = int(len(rlist)*legshare)
    if anzwin==0:anzwin=1
    if anz + anzwin > len(rlist)-1:
        anzwin = len(rlist)-1-anz
    minscore = sorted(rlist)[anz]
    maxscore = sorted(rlist)[-anzwin]
    rl = []
    wl = []
    for i in range(len(rlist)):
        if rlist[i] < minscore or rlist[i]==-1000:
            rl.append(i)
        if rlist[i] >= maxscore:
            wl.append(i)
            print('win:',i,rlist[i])
    return [rl,wl]


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

def initialize(pfile,mfile,attvar,verbose=1):
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
    pdic = create_pdic(pdata,mdic.keys(),attvar,verbose=verbose)
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


def create_pdic(data,media,attvar="Einstellung_Pro1", verbose = 1):
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
            v = float(data[attvar][i]) ##Remove agents of whom we do not know the attitude
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
               
            pdic[ident]['Einst'] = pdic[ident][attvar] #Initalize the dynamic attitude
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


def simulate(pdic, mdic, steps, param, benchvar="Einstellung_Pro2"):
    ## Actual simulation, using the population, media environment, points in time, and parameters
    results = []
    ts = time.time()
    outsim = {}
    outsimvar=['Initial']
    outsim['Initial']=vectorize(pdic,'Einst')
    b_cert = param['coeff_e'] ## Use the coefficients directly
    b_reli = param['coeff_a']

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

        for p in pdic.keys():
            pdic[p]['Einst'] = pdic[p]['Einst']+pdic[p]['dEinst']
        r = calc_identity(pdic,'Einst',benchvar)
        verb('Mean square sum on day: '+str(d)+' = '+str(1.0/r),verbose=3)

        verb('.',nl=0,verbose=2)
        results.append(r)
        outsim[str(d)]=vectorize(pdic,'Einst')
        outsimvar.append(str(d))
        
    pearson_r = calc_pearson(pdic,'Einst',benchvar)

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
    return [results[-1],pearson_r]



####################################################
###
### Main program
###
####################################################

## Set the basic settings for the evolutionary algorithm and bootstrapping

global g_settings
g_settings = {}
manualset = {'Gen_Size':30,
             'Max_Gen':1,
             'Max_Memory':60,
             'Selection':.5,
             'Legends':.1,
             'Mutation': .2,
             'N_Bootstrap':2000,
             'S_Bootstrap':600}
init_settings('G_Settings.json',manualset,force=1)

## If, at any later time during the evolution of parameter sets, you wish to change
## one of these values, you may edit them in the 'G_Settings.json' file.
## If the distribution of a coefficient is to be changed, use the parameter: 'Overwrite':{'coeff_e':{'M':1.5, 'SD':0.01}}
## (In this case to overwrite the distribution of coeff_e with a mean of 1.5 and a standard deviation of 0.01


##Set prior distributions (m and sd) for each parameter
##pset = {'coeff_a':(0.0,.1,'norm'), ##Effect of Media Reliance
##        'coeff_b':(0.0,.1,'norm'), ##Intercept: Media Impact
##        'coeff_c':(0.0,.1,'norm'), ##Intercept: Social Impact
##        'coeff_d':(2.0,.1,'normp'), ##Ratio of Spatial vs. Ideological distance
##        'coeff_e':(0.0,.1,'norm')} ## Effect of Attitude certainty

pset = {'coeff_a':(0,.01,'norm'), ##Effect of Media Reliance
        'coeff_b':(0,.01,'norm'), ##Intercept: Media Impact
        'coeff_c':(0,.01,'norm'), ##Intercept: Social Impact
        'coeff_d':(1.0,.03,'normp'), ##Ratio of Spatial vs. Ideological distance
        'coeff_e':(0,.01,'norm')} ## Effect of Attitude certainty

prior = evol_initialize(pset,g_settings['Gen_Size']) ## Initiate the prior distribution of parameter sets.


## Set the boundary conditions for the simulation
phase = 1
if phase == 1: ## Distinguish between predicting the interval between wave 1&2 and wave 2&3
    simdates=[38927,38934,38941,38948,38955,38962] #Dates for Phase 1
    var_e1 = 'Einstellung_Pro1'
    var_e2 = 'Einstellung_Pro2'
elif phase == 2:
    simdates=[38962,38969,38976,38983] #Dates for Phase 2
    var_e1 = 'Einstellung_Pro2'
    var_e2 = 'Einstellung_Pro3'

## Set the names of the report files    
outfile1 = "SimResult_within_full.txt"        ## Complete collection of simulation results for each parameter set
outfile2 = "SimResult_within.txt"             ## Summary per generation, providing M, SD and 95% CI for each parameter
summary_file = 'Summary_within.txt'           ## Summary after maximal number of generations, including cross-validation with test data

res_outfile = 1 ## Setting to reset or retain outfile1 and 2: 1: New Simulation output, 0: Append to previous output
res_pset = 0 ## Setting to reset the prior distribution after each bootstrapping sample: 1: reset / 0: use the posterior from last sample

bootstep = 0
while bootstep < g_settings['N_Bootstrap']:
    bootstep = bootstep + 1
    ##Bootstrapping loop
    write_samples('Survey_Data.dat','train_sample.dat','test_sample.dat',g_settings['S_Bootstrap'])
    if res_outfile == 1:
        tr = open('tmp_w1_bs.txt','w')
        for p in sorted(pset.keys()):
            tr.write(str(p)+'\t')
        for s in simdates:
            tr.write('Day_'+str(s)+'\t')
        tr.write('\n')
        tr.close()

    if res_pset == 1:
        prior = evol_initialize(pset,g_settings['Gen_Size'])

    generation = 0
    while generation < g_settings['Max_Gen']:
        ## Loop of the evolutionary algorithm
        
        verb('\n\n####################\n##### Generation: '+str(generation+1)+' of '+str(g_settings['Max_Gen'])+'\n####################')
        prior['Results'] = []
        init_settings('G_Settings.json',g_settings) ##Re-initialize settings. Probably overwrite old settings with new ones.
        for i in range(g_settings['Gen_Size']): ## Repeat the simulation for each parameter set
            dr2_list = []
            try:
                param = {}
                for p in pset.keys():
                    param[p] = prior[p]['Curr'][i]  ## Set the parameters to the parameter set to be used

                dics = initialize('train_sample.dat','Content_Data.dat',var_e1,verbose=0) ## Initialize the simulation
                pcorr_init_r = calc_pearson(dics[0],var_e1,var_e2) ## Get the initial correlation of attitudes
                pcorr_init_id = calc_identity(dics[0],var_e1,var_e2) ## Get the initial agreement of attitudes
                verb('\nInitial inverted squaresum: '+str(pcorr_init_id),verbose=1)
                verb('\nParameters: '+str(param),verbose=1)
                simresult = simulate(dics[0],dics[1],simdates,param,var_e2) ## Call the simulation and return the final agreement
                prior['Results'].append(simresult[0]) ## Notify the evolutionary algorithm of the result of this simulation
            except Exception as fehler:
                ##Emergency Error catch. The computation will not end because of strange errors. It just notifies the user.
                ##One reason for this error may be that the number of parameter sets per generation 'Gen_Size' was changed
                ##when a prior with a different generation size was already initialized. Will resolve itself on its own
                ##at the end of this generation when new parameter sets are initialized.
                prior['Results'].append(-1000)
                verb(str(fehler),0)
                verb('ERROR: Very strange error. Could not simulate individual. Dropping it with result = -1000')

##            inex = pcorr_init_r**2
##            inex2 = simresult[1]**2
##            dr2 = (simresult[1]**2-pcorr_init_r**2)*100
            inex = 1/pcorr_init_id ## Mean square deviation between the waves
            exp = 1/simresult[0] ## Mean square deviation after simulation
            dr2 = (inex-exp)/inex*100 ## Proportional reduction of error: Percent of explained variance by simulation
            dr2_list.append(dr2)
            
            verb('delta R2 (explained variance by simulation): '+"{0:2.2f}%".format(dr2),verbose=1)
            
            if pcorr_init_id < prior['Results'][-1]:
                verb('Result for Individual['+str(i)+']: '+str(i+1)+' from '+str(g_settings['Gen_Size'])+' (Gen: '+str(generation+1)+'/'+str(g_settings['Max_Gen'])+'): '+str(prior['Results'][-1])+'**',verbose=0)
                verb(niceprint_tree(param),verbose=1)
            else:
                verb('Result for Individual['+str(i)+']: '+str(i+1)+' from '+str(g_settings['Gen_Size'])+' (Gen: '+str(generation+1)+'/'+str(g_settings['Max_Gen'])+'): '+str(prior['Results'][-1]),verbose=0)
                verb(niceprint_tree(param),verbose=1)

        ## Inform on final result for this generation of the evolutionary algorithm
        
        verb('\nInitial agreement was: '+str(pcorr_init_id),verbose=0)
        special_ind = evol_fail(prior['Results'],g_settings['Selection'],g_settings['Legends'])
        rem_individuals = special_ind[0]
        win_individuals = special_ind[1]
        verb('Removing individuals from herd: '+str(rem_individuals), nl=0,verbose=0)
        verb('Kepping individuals for breeding: '+str(win_individuals), nl=0,verbose=0)
        verb('\nLowest result: '+str(min(prior['Results'])),verbose=0)

        resline = time.ctime()+'\t'+str(pcorr_init_id)+'\t'+str(c_mittel(prior['Results']))+'\t'+str(c_sdev(prior['Results']))+'\t'+str(c_mittel(dr2_list))
        evol_write_result(prior,outfile1)

        prior = evol_update(prior,special_ind,g_settings['Gen_Size'],g_settings['Max_Memory'],g_settings['Mutation'],outfile2)
        generation = generation + 1

    ## End of the evolutionary loop after the specified number of generations.
    ## The result is a dictionary only containing the optimal fits (prior).


    ## Cross-Validate the solution with the test-cases:

    verb('\nEvaluating optimal parameters in test dataset..',verbose=0)
    
    mean_params = {}
    for p in pset.keys():
        param[p] = c_mittel(prior[p]['Prev']) ## Set parameters to the mean of the optimal solutions

    dics = initialize('test_sample.dat','Content_Data.dat',var_e1,verbose=0)
    pcorr_init_id = calc_identity(dics[0],var_e1,var_e2)
    simresult = simulate(dics[0],dics[1],simdates,param,var_e2)
    
    inex = 1/pcorr_init_id ## Mean square deviation between the waves
    exp = 1/simresult[0] ## Mean square deviation after simulation
    dr2 = (inex-exp)/inex*100 ## Proportional reduction of error: Percent of explained variance by simulation
    print(inex,exp,dr2)
                
    resline = resline+'\t'+str(dr2)
    verb('\n\n >> Delta R2 (explained variance by simulation of test data): '+"{0:2.2f}%".format(dr2),verbose=0)
    verb('(Mean Delta R2 in training: '+"{0:2.2f}%)\n\n".format(c_mittel(dr2_list)),verbose=0)


    ## Output of Results

    if bootstep == 1:
        fullresult = open(summary_file,'w')
        fullresult.write('TS\tInitial_Agreement\tFinal_Agreement\tSD_Final_Agreement\tDRSQ_Training\tDRSQ_Test')
        for p in sorted(prior.keys()):
            fullresult.write('\tM_'+p+'\tSD_'+p+'\tLow_'+p+'\tHigh_'+p)
        fullresult.write('\n')
        fullresult.close()

    fullresult = open(summary_file,'a')
    fullresult.write(resline)

    for p in sorted(prior.keys()):
        m=c_mittel(prior[p]['Prev'])
        sd=c_sdev(prior[p]['Prev'])
        low=c_low(prior[p]['Prev'])
        hig=c_hig(prior[p]['Prev'])
        verb(p+': M='+str(m)+' ; SD='+str(sd),verbose=0)
        fullresult.write('\t'+str(m)+'\t'+str(sd)+'\t'+str(low)+'\t'+str(hig))

    fullresult.write('\n')
    fullresult.close()
    time.sleep(5)

## End of bootstrapping. The summary file now includes the bootstrapping results for further inspection.

