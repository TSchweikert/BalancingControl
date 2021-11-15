#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 12:16:04 2021

@author: timo
"""

import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import scipy.stats as sc
import seaborn as sns
from pandas.plotting import scatter_matrix
import distributions as analytical_dists
from statsmodels.graphics.gofplots import qqplot


"""Einlesen der Daten, Formen des Dataframes und der fehlende Werte"""

header = []
daten = []

pfad = '/home/baemm/Coding/BalancingControl/Daten_Matlab/Inferiert_Mensch'
dateien = os.listdir(pfad)
dateien = sorted(dateien)

for datei in dateien:
    with open(f'{pfad}/{datei}', 'r') as zu_lesen:
        reader = csv.reader(zu_lesen, delimiter=',')
        header = next(reader)
        daten.extend([row for row in reader])

ergebnis_Mensch  = pd.DataFrame(data=daten, dtype=np.float32)
ergebnis_Mensch = ergebnis_Mensch.rename(columns=(dict(zip(ergebnis_Mensch.columns,header))))
del ergebnis_Mensch[""]

LOSS_Mensch = pd.DataFrame()
a = []
i=0
pfad = '/home/baemm/Coding/BalancingControl/Daten_Matlab/LOSS_Mensch'
dateien = os.listdir(pfad)
dateien = sorted(dateien)


for datei in dateien:
    with open(f'{pfad}/{datei}', 'r') as zu_lesen:
        reader = pd.read_csv(zu_lesen, delimiter=',')
        a =  reader
        a = pd.DataFrame(a["0"])
        LOSS_Mensch[str(i)] = a["0"]
    i = i+1

"""Modus und Mean der Verteilung berechnen - lambda pi"""
# modus (a-1)/(a+b-2) bei a und b >1

np_data = ergebnis_Mensch.to_numpy()
a = np_data[:,0]
b = np_data[:,1]

Modus = (a -1) / (a + b -2)
ergebnis_Mensch['Modus_pi'] = Modus 

#mean a/(a+b)
Mean = (a) / (a + b)
ergebnis_Mensch['Mean_pi'] = Mean

"""Modus und Mean der Verteilung berechnen - lambda r"""
#(a-1)/(a+b-2)

a = np_data[:,2]
b = np_data[:,3]

Modus_rr = (a -1) / (a + b -2)
ergebnis_Mensch['Modus_r'] = Modus_rr

#mean a/(a+b)
Mean_rr = (a) / (a + b)
ergebnis_Mensch['Mean_r'] = Mean_rr

"""Mean und Modus Gamma Verteilung - decision temperature"""
#modus = (a-1)/b bei a >1
a = np_data[:,4]
b = np_data[:,5]
    
Modus_dtt = (a - 1 ) / (b)
ergebnis_Mensch['Modus_dt'] = Modus_dtt

#mean = a/b
Mean_dtt  = (a/ b)
ergebnis_Mensch['Mean_dt'] = Mean_dtt

ergebnis_Mensch_alle = ergebnis_Mensch
#%%
def eva(x,v2):
    """x = DataFrame; v1 = Variable als string; v2 = Proband des Interesse als int
    Rückgabe: Counts der Variable ueber alle Probanden"""
    #Tabelle = x[v1].value_counts(sort=True)
    
    #print("Häufigkeit für " +v1+ " in Zahlen ueber alle Probanden:\n\n", Tabelle)
    #print("Häufigkeit für Proband " +str(v2)+ " der Variable "+v1+":\n",x.loc[v2,v1])
    print("Alle Werte des Probanden_" +str(v2)+":\n Modus_dt",x.loc[v2,"Modus_dt"],"\n Mean_dt",x.loc[v2,"Mean_dt"],"\n Modus_pi",x.loc[v2,"Modus_pi"],"\n Mean_pi",x.loc[v2,"Mean_pi"],"\n Modus_r",x.loc[v2,"Modus_r"],"\n Mean_r",x.loc[v2,"Mean_r"],"\n mf_score",x.loc[v2,"mf_score"],'\n')
    
    # Histogramm und Tabellen
    #plt.figure()
    #plt.title("Histogramm für " +v1)
    #plt.hist(x[v1])
    #plt.show()
 
        
    x_lamb = np.arange(0.01,1.,0.01)
    x_dec_temp = np.arange(0.01,10.,0.01)
    
    y_lamb_pi = analytical_dists.Beta(x_lamb, x.loc[v2,"alpha_lamb_pi"], x.loc[v2,"beta_lamb_pi"])
    y_lamb_r = analytical_dists.Beta(x_lamb, x.loc[v2,"alpha_lamb_r"], x.loc[v2,"beta_lamb_r"])
    y_dec_temp = analytical_dists.Gamma(x_dec_temp, concentration=x.loc[v2,"concentration_dec_temp"], rate=x.loc[v2,"rate_dec_temp"])

    
    # Elbo + Verteilung Proband
    plt.figure(figsize=(14, 8))
    #plt.title("ELBO")
    plt.subplot(2,2,1)
    plt.plot(LOSS_Mensch[str(v2)])
    plt.ylabel("ELBO")
    plt.xlabel("iteration")
    
    plt.subplot(2,2,2)
    #plt.title("Beta_pi")
    plt.plot(x_lamb,y_lamb_pi)
    plt.xlim([0.01-0.01,0.99+0.01])
    plt.xlabel("forgetting rate prior policies: $\\lambda_{\pi}$")
    plt.ylabel("Beta_pi")
    
    plt.subplot(2,2,3)
    #plt.title("Beta_r")
    plt.plot(x_lamb,y_lamb_r)
    plt.xlim([0.01-0.01,0.99+0.01])
    plt.xlabel("forgetting rate reward probabilities: $\\lambda_{r}$")
    plt.ylabel("Beta_r")
    
    plt.subplot(2,2,4)
    #plt.title("Gamma_dt")
    plt.plot(x_dec_temp,y_dec_temp)
    np.arange(0.01-0.01,9.99+0.01)
    plt.xlabel("decision temperature: $\\gamma$")
    plt.ylabel("Gamma_dt")
    
    plt.suptitle('Proband '+str(v2),fontweight ="bold")
    plt.tight_layout()
    plt.show()
    
    
    
def histfit(x,c):    
    #plt.hist(ergebnis_Mensch['Modus_r'])
    _, bins, _ = plt.hist(c[x],bins=30,density = True,ec='black',linewidth=0.5)
    mu, sigma = sc.norm.fit(c[x])
    best_fit_line = sc.norm.pdf(bins, mu, sigma)
    plt.plot(bins, best_fit_line,'r',linewidth=2)
    plt.xlabel(x)
    plt.ylabel('Frequency')
    return    
#%%    
def corri(x,v1,v2):
    #'Modus_dt' / 'Mean_dt'
    corr, pvalue = sc.pearsonr(x[v1], x[v2])
    print("Korrelationskoeffizient für "+v1+" und "+v2+":", corr)
    print("P-Value für "+v1+" und "+v2+":",pvalue)
    print("P-Value komplett für "+v1+" und "+v2+":","{:0.30f}".format(pvalue))
    
    # creating X-Y Plots With a Regression Line
    # slope, intersept, and correlation coefficient calculation 
    slope, intercept, r, p, stderr = sc.linregress(x[v1], x[v2])
    line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}'
    # plotting
    fig, ax = plt.subplots(figsize = (14,8))
    ax.plot(x[v1], x[v2], linewidth=0, marker='s', label='Data points')
    ax.plot(x[v1], intercept + slope * x[v1], label=line)
    ax.set_xlabel(v1)
    ax.set_ylabel(v2)
    ax.legend(facecolor='white')
    fig.suptitle("Scatterplot "+v1+" und "+v2,fontweight = "bold")
    plt.tight_layout()
    plt.show()
    print("\n\n")
    
corri(ergebnis_Mensch_alle,'Modus_dt','Mean_dt')
corri(ergebnis_Mensch_alle,'Modus_pi','Mean_pi')
corri(ergebnis_Mensch_alle,'Modus_r','Mean_r')    


# Korrelation Modus und Mean 
#Weniger Variablen
df = ergebnis_Mensch_alle[['Modus_dt','Modus_r','Modus_pi','Mean_dt','Mean_r','Mean_pi']]

#Gleiche wie unten mit relevanteren Variablen
scatter_matrix(df, figsize=(14,8)) #sieht ein wenig umständlich aus daher das darüber
plt.suptitle('Scattermatrix über alle Probanden',fontweight ="bold")
plt.tight_layout()
plt.show() 

plt.figure(figsize=(14, 8))
plt.subplot(2,3,1)
histfit('Modus_dt',ergebnis_Mensch_alle)

plt.subplot(2,3,2)
histfit('Modus_r',ergebnis_Mensch_alle)

plt.subplot(2,3,3)
histfit('Modus_pi',ergebnis_Mensch_alle)

plt.subplot(2,3,4)
histfit('Mean_dt',ergebnis_Mensch_alle)

plt.subplot(2,3,5)
histfit('Mean_r',ergebnis_Mensch_alle)

plt.subplot(2,3,6)
histfit('Mean_pi',ergebnis_Mensch_alle)
plt.suptitle('Histogramm über alle Probanden',fontweight ="bold")
plt.tight_layout()
plt.show()

#%%  
import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import scipy.stats as sc
import seaborn as sns
from pandas.plotting import scatter_matrix
import distributions as analytical_dists
from mat4py import loadmat

#Anpassen an die Anzahl der Probanden = 188
n=188

mfmb_scores = np.zeros(shape=(n,2))
pfad = '/home/baemm/Coding/BalancingControl/Daten_Matlab/MBMF_Scores'
dateien = os.listdir(pfad)
dateien = sorted(dateien)
i=0


for datei in dateien:
        reader = loadmat('/home/baemm/Coding/BalancingControl/Daten_Matlab/MBMF_Scores/'+ datei)
        amf= reader['Scores_'][0]
        amb = reader['Scores_'][1]

        mfmb_scores[i,0]= amf
        mfmb_scores[i,1] = amb
        i=i+1
ergebnis_Mensch_alle['mf_score'] = mfmb_scores[:,0]
ergebnis_Mensch_alle['mb_score'] = mfmb_scores[:,1]

plt.subplot(1,2,1)
histfit('mf_score',ergebnis_Mensch_alle)
plt.subplot(1,2,2)
histfit('mb_score',ergebnis_Mensch_alle)
plt.suptitle('Histogramm über alle Probanden',fontweight ="bold")
plt.tight_layout()
plt.show()
#%%
#Beta1,Beta2, alpha1,alpha2,lambda,w,rep
#Spalte 1:7 unstransformierte Werte; Spalte 8:14 Transformiert 
# par.EEt(1:2,:) =       exp( par.EE(1:2,:) ); % beta   = exp(x(1:2));
# par.EEt(3:4,:) = 1./(1+exp(-par.EE(3:4,:))); % alpha  = 1./(1+exp(-x(3:4)));
# par.EEt(  5,:) = 1./(1+exp(-par.EE(  5,:))); % lambda = 1./(1+exp(-x(5)));
# par.EEt(  6,:) = 1./(1+exp(-par.EE(  6,:))); % w      = 1./(1+exp(-x(6)));
# par.EEt(  7,:) =            par.EE(  7,:)  ; % rep    = x(7)*eye(2);

import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import scipy.stats as sc
import seaborn as sns
from pandas.plotting import scatter_matrix
import distributions as analytical_dists
from mat4py import loadmat

Probanden_para = np.zeros(shape=(n,14))
pfad = '/home/baemm/Coding/BalancingControl/Daten_Matlab/Probanden_Parameter'
dateien = os.listdir(pfad)
dateien = sorted(dateien)
i=0


for datei in dateien:
        reader = loadmat('/home/baemm/Coding/BalancingControl/Daten_Matlab/Probanden_Parameter/'+ datei)
        Probanden_para[i,:]=reader['Probanden_parameter'][0:14]
        i = i+1
ergebnis_Mensch_alle['Beta1'] = Probanden_para[:,0]
ergebnis_Mensch_alle['Beta2'] = Probanden_para[:,1]     
ergebnis_Mensch_alle['Alpha1'] = Probanden_para[:,2]
ergebnis_Mensch_alle['Alpha2'] = Probanden_para[:,3]  
ergebnis_Mensch_alle['lambda'] = Probanden_para[:,4]
ergebnis_Mensch_alle['w'] = Probanden_para[:,5]  
ergebnis_Mensch_alle['rep'] = Probanden_para[:,6]
ergebnis_Mensch_alle['Beta1_trans'] = Probanden_para[:,7]  
ergebnis_Mensch_alle['Beta2_trans'] = Probanden_para[:,8]
ergebnis_Mensch_alle['Alpha1_trans'] = Probanden_para[:,9]  
ergebnis_Mensch_alle['Alpha2_trans'] = Probanden_para[:,10]
ergebnis_Mensch_alle['lambda_trans'] = Probanden_para[:,11]
ergebnis_Mensch_alle['w_trans'] = Probanden_para[:,12]
ergebnis_Mensch_alle['rep_trans'] = Probanden_para[:,13]       

plt.figure(figsize=(14, 8))
plt.subplot(3,3,1)
histfit('Beta1_trans',ergebnis_Mensch_alle)

plt.subplot(3,3,2)
histfit('Beta2_trans',ergebnis_Mensch_alle)

plt.subplot(3,3,3)
histfit('Alpha1_trans',ergebnis_Mensch_alle)

plt.subplot(3,3,4)
histfit('Alpha2_trans',ergebnis_Mensch_alle)

plt.subplot(3,3,5)
histfit('lambda_trans',ergebnis_Mensch_alle)

plt.subplot(3,3,6)
histfit('w_trans',ergebnis_Mensch_alle)

plt.subplot(3,3,7)
histfit('rep_trans',ergebnis_Mensch_alle)
plt.suptitle('Histogramm über alle Probanden',fontweight ="bold")
plt.tight_layout()
plt.show()


# Korrelation 
##Beta1_trans,Beta2_trans, Alpha1_trans,Alpha2_trans,lambda_trans,w_trans,rep_trans
#df = ergebnis_Mensch_alle[['Modus_dt','Modus_r','Modus_pi','Mean_dt','Mean_r','Mean_pi','Beta1_trans','Beta2_trans', 'Alpha1_trans','Alpha2_trans','lambda_trans','w_trans','rep_trans','mb_score','mf_score']]
#df = ergebnis_Mensch_alle[['Modus_dt','Modus_r','Modus_pi','Mean_dt','Mean_r','Mean_pi','Beta1','Beta2', 'Alpha1','Alpha2','lambda','w_trans','rep','mb_score','mf_score']]
df = ergebnis_Mensch_alle[['Modus_dt','Modus_r','Modus_pi','Beta1','Beta2', 'Alpha1','Alpha2','lambda','w_trans','rep','mb_score','mf_score']]

scatter_matrix(df, figsize=(14,8)) #sieht ein wenig umständlich aus daher das darüber
plt.suptitle('Scattermatrix über alle Probanden',fontweight ="bold")
plt.tight_layout()
plt.show() 

# QQ Plot
for i in ['Modus_dt', 'Modus_pi', 'Modus_r','mf_score','mb_score','w_trans']:
    qqplot(ergebnis_Mensch_alle[i], line='s')
    plt.suptitle('QQ-Plot über alle Probanden von Variable '+i,fontweight ="bold")
    plt.tight_layout()
    plt.show()

    stat, p = sc.shapiro(ergebnis_Mensch_alle[i])
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
    	print('Sample looks Gaussian for ' +i+ ' (fail to reject H0)\n')
    else:
    	print('Sample does not look Gaussian for ' +i+ ' (reject H0)\n')
        
#for i in range(0,ergebnis_Mensch_alle.shape[0]):
#    eva(ergebnis_Mensch_alle,i)
#%%
#df = ergebnis_Mensch_alle[['Modus_dt','Modus_r','Modus_pi','Beta1','Beta2', 'Alpha1','Alpha2','lambda','w_trans','rep','mb_score','mf_score']]
#df = ergebnis_Mensch_alle[['Modus_dt','Modus_r','Modus_pi','Mean_dt','Mean_r','Mean_pi','Beta1_trans','Beta2_trans', 'Alpha1_trans','Alpha2_trans','lambda_trans','w_trans','rep_trans','mb_score','mf_score']]
df = ergebnis_Mensch_alle[['Modus_dt','Modus_r','Modus_pi','Mean_dt','Mean_r','Mean_pi','Beta1','Beta2', 'Alpha1','Alpha2','lambda','w_trans','rep','mb_score','mf_score']]
#df = df = ergebnis_Mensch_alle[['Modus_dt','Modus_r','Modus_pi','Mean_dt','Mean_r','Mean_pi']]
def corr_sig(df=None):
    p_matrix = np.zeros(shape=(df.shape[1],df.shape[1]))
    for col in df.columns:
        for col2 in df.drop(col,axis=1).columns:
            _ , p = sc.pearsonr(df[col],df[col2])
            p_matrix[df.columns.to_list().index(col),df.columns.to_list().index(col2)] = p
    return p_matrix

p_values = corr_sig(df)
mask = np.invert(np.tril(p_values<0.05))



def plot_cor_matrix(corr, mask=None):
    f, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, ax=ax,
                mask=mask,
                # cosmetics
                annot=True, vmin=-1, vmax=1, center=0, square=True, 
                cmap='coolwarm', linewidths=0.1, linecolor='black', cbar_kws={'orientation': 'vertical'})
    #sns.heatmap(corr, mask=mask, cmap=sns.diverging_palette(220, 10, as_cmap=True),
     #              square=True, annot=True, ax=ax)
# Plotting without significance filtering
#corr = df.corr()
#mask = np.triu(corr)
#plot_cor_matrix(corr,mask)
#plt.show()

# Plotting with significance filter
corr = df.corr()                            # get correlation
p_values = corr_sig(df)                     # get p-Value
mask = np.invert(np.tril(p_values<0.05))    # mask - only get significant corr
plot_cor_matrix(corr,mask)
plt.title("Heatmap mit allen Probanden",fontweight ="bold")



for i in range(0,ergebnis_Mensch_alle.shape[0]):
    eva(ergebnis_Mensch_alle,i)
    
#%%
#Entfernen aller nicht konvergierten
raus = 28
ergebnis_Mensch_alle_ohne = ergebnis_Mensch_alle.drop([6,12,14,21,29,34,53,57,70,77,85,88,97,100,111,115,120,135,139,145,150,152,156,158,162,166,178,181])
nicht_konv= (raus*100)/188
print('Probanden die nicht konvergiert sind: ',nicht_konv,'%')

#entfernen aller Modus_dt < 1.0
#ergebnis_Mensch_alle = ergebnis_Mensch_alle[ergebnis_Mensch_alle['Modus_dt'] > 1.0]


#df = ergebnis_Mensch_alle[['Modus_dt','Modus_r','Modus_pi','Beta1','Beta2', 'Alpha1','Alpha2','lambda','w_trans','rep','mb_score','mf_score']]
#df = ergebnis_Mensch_alle[['Modus_dt','Modus_r','Modus_pi','Mean_dt','Mean_r','Mean_pi','Beta1_trans','Beta2_trans', 'Alpha1_trans','Alpha2_trans','lambda_trans','w_trans','rep_trans','mb_score','mf_score']]
df = ergebnis_Mensch_alle_ohne[['Modus_dt','Modus_r','Modus_pi','Mean_dt','Mean_r','Mean_pi','Beta1','Beta2', 'Alpha1','Alpha2','lambda','w_trans','rep','mb_score','mf_score']]
#df = df = ergebnis_Mensch_alle[['Modus_dt','Modus_r','Modus_pi','Mean_dt','Mean_r','Mean_pi']]

corr = df.corr()                            # get correlation
p_values = corr_sig(df)                     # get p-Value
mask = np.invert(np.tril(p_values<0.05))    # mask - only get significant corr
plot_cor_matrix(corr,mask)
plt.title("Heatmap nur mit konvergierten Probanden",fontweight ="bold")

# Modus ist weder für Gamma noch Beta verwendbar da Beta = a,b > 1 und Gamme a >1
ergebnis_Mensch = ergebnis_Mensch_alle.drop([6,12,14,21,29,34,53,57,70,77,85,88,97,100,111,115,120,135,139,145,150,152,156,158,162,166,178,181])
ergebnis_Mensch = ergebnis_Mensch[ergebnis_Mensch['concentration_dec_temp'] > 1.0]
ergebnis_Mensch = ergebnis_Mensch[ergebnis_Mensch['alpha_lamb_pi'] > 1.0]
ergebnis_Mensch = ergebnis_Mensch[ergebnis_Mensch['beta_lamb_pi'] > 1.0]
ergebnis_Mensch = ergebnis_Mensch[ergebnis_Mensch['alpha_lamb_r'] > 1.0]
ergebnis_Mensch = ergebnis_Mensch[ergebnis_Mensch['beta_lamb_pi'] > 1.0]
#ergebnis_Mensch = ergebnis_Mensch[ergebnis_Mensch['Modus_dt'] > 1.0]
    
corri(ergebnis_Mensch,'Modus_dt','Mean_dt')
corri(ergebnis_Mensch,'Modus_pi','Mean_pi')
corri(ergebnis_Mensch,'Modus_r','Mean_r')      

df = ergebnis_Mensch[['Modus_dt','Modus_r','Modus_pi','Mean_dt','Mean_r','Mean_pi','Beta1','Beta2', 'Alpha1','Alpha2','lambda','w_trans','rep','mb_score','mf_score']]
corr = df.corr()                            # get correlation
p_values = corr_sig(df)                     # get p-Value
mask = np.invert(np.tril(p_values<0.05))    # mask - only get significant corr
plot_cor_matrix(corr,mask)
plt.title("Heatmap nur mit konvergierten Probanden und korrektem Modus",fontweight ="bold")


# # QQ Plot
# for i in ['Mean_dt', 'Mean_pi', 'Mean_r','mf_score','mb_score','w_trans']:
#     qqplot(ergebnis_Mensch[i], line='s')
#     plt.suptitle('QQ-Plot über alle Probanden von Variable '+i,fontweight ="bold")
#     plt.tight_layout()
#     plt.show()

#     stat, p = sc.shapiro(ergebnis_Mensch_alle[i])
#     print('Statistics=%.3f, p=%.3f' % (stat, p))
#     # interpret
#     alpha = 0.05
#     if p > alpha:
#     	print('Sample looks Gaussian for ' +i+ ' (fail to reject H0)\n')
#     else:
#     	print('Sample does not look Gaussian for ' +i+ ' (reject H0)\n') 
#%%

#sns.lmplot(x="Modus_dt", y="Mean_dt", data=ergebnis_Mensch) 

#sns.lmplot(x="Modus_pi", y="Mean_pi", data=ergebnis_Mensch) 

#sns.lmplot(x="Modus_r", y="Mean_r", data=ergebnis_Mensch) 
ergebnis_Mensch = ergebnis_Mensch_alle.drop([6,12,14,21,29,34,53,57,70,77,85,88,97,100,111,115,120,135,139,145,150,152,156,158,162,166,178,181])
#ergebnis_Mensch = ergebnis_Mensch[ergebnis_Mensch['Modus_dt'] > 1.0]


plt.figure(figsize=(14, 8))
plt.subplot(1,3,1)
histfit('Mean_dt',ergebnis_Mensch)

plt.subplot(1,3,2)
histfit('Mean_pi',ergebnis_Mensch)

plt.subplot(1,3,3)
histfit('Mean_r',ergebnis_Mensch)


plt.suptitle('Histogram',fontweight ="bold")
plt.tight_layout()
plt.show()


for i in ['Mean_dt', 'Mean_pi', 'Mean_r']:

    stat, p = sc.shapiro(ergebnis_Mensch_alle[i])
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
     	print('Sample looks Gaussian for ' +i+ ' (fail to reject H0)\n')
    else:
     	print('Sample does not look Gaussian for ' +i+ ' (reject H0)\n') 

#%%


def plot_cor_matrix(corr, mask=None):
    f, ax = plt.subplots(figsize=(14,10))
    sns.heatmap(corr, ax=ax,
                mask=mask,
                # cosmetics
                annot=True, vmin=-1, vmax=1, center=0, square=True, 
                cmap='coolwarm', linewidths=0.01, linecolor='black', cbar_kws={'orientation': 'vertical'})


df = ergebnis_Mensch[['Mean_dt','Mean_r','Mean_pi','Beta1','Beta2', 'Alpha1','Alpha2','lambda','w','rep','mb_score','mf_score']]
df = df.rename(columns={"mb_score": "mb-sc", "mf_score": "mf-sc"})
corr = df.corr()                            # get correlation
p_values = corr_sig(df)                     # get p-Value
mask = np.invert(np.tril(p_values<0.0007))    # mask - only get significant corr
plot_cor_matrix(corr,mask)
plt.title("Correlation Matrix",fontweight ="bold")




raus = 28
nicht_konv= (raus*100)/188
print('Probanden die nicht konvergiert sind: ',nicht_konv,'%')