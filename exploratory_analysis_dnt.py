# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ### Initial setup

# %%
#Import libraries
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
print("Packages uploaded successfully!")


# %%
#Read dataset
data = pd.read_csv('./dnt_machine_readable.csv')


# %%
#Check first five rows
data.head(5)

# %% [markdown]
# ### Distribution of categorical variables

# %%
#DNT
replace_values = {0 : 'Negative/Safe', 1 : 'Positive'}
dnt = data.replace({"DNT": replace_values})
#sns.set_palette("Set3") #or pastel
#sns.set_palette("Reds")
my_palette = ["#e74c3c", "#2ecc71"]
sns.set_palette(sns.color_palette(my_palette))
ax = sns.countplot(x="DNT", data=dnt)
ax.set(ylabel='Number of compounds', xlabel=None, title ='DNT Classification')
sns.despine(right=True, bottom = True)


# %%
#BBB
replace_values_bbb = {0 : 'Unpermeable', 1 : 'Permeable'}
bbb = data.replace({"BBB": replace_values_bbb})
my_palette = ["#e74c3c", "#2ecc71"]
sns.set_palette(sns.color_palette(my_palette))
ax = sns.countplot(x="BBB", data=bbb)
ax.set(ylabel='Number of compounds', xlabel=None, title ='Blood-brain-barrier Permeability')
sns.despine(right=True, bottom = True)


# %%
#Pgp_inhibition
replace_values_pgp_inh = {1 : 'Inhibitor', 0 : 'Non-inhibitor'}
pgp_inh = data.replace({"Pgp_inhibition": replace_values_pgp_inh})
my_palette = ["#e74c3c", "#2ecc71" ]
sns.set_palette(sns.color_palette(my_palette))
ax = sns.countplot(x="Pgp_inhibition", data=pgp_inh, order = ["Inhibitor", "Non-inhibitor"])
ax.set(ylabel='Number of compounds', xlabel=None, title ='Inhibition of P-glycoprotein')
sns.despine(right=True, bottom = True)


# %%
#Pgp_substrate
replace_values_pgp_subs = {0 : 'Non-substrate', 1 : 'Substrate'}
pgp_subs = data.replace({"Pgp_substrate": replace_values_pgp_subs})
my_palette = ["#e74c3c","#2ecc71"]
sns.set_palette(sns.color_palette(my_palette))
ax = sns.countplot(x="Pgp_substrate", data=pgp_subs, order = ["Substrate", "Non-substrate"])
ax.set(ylabel='Number of compounds', xlabel=None, title ='Substrate to P-glycoprotein')
sns.despine(right=True, bottom = True)


# %%
#Pgp_active
replace_values_pgp_act = {0 : 'Inactive', 1 : 'Active'}
pgp_act = data.replace({"Pgp_active": replace_values_pgp_act})
my_palette = ["#e74c3c", "#2ecc71"]
sns.set_palette(sns.color_palette(my_palette))
ax = sns.countplot(x="Pgp_active", data=pgp_act, order = ["Active", "Inactive"])
ax.set(ylabel='Number of compounds', xlabel=None, title ='Activity againts P-glycoprotein')
sns.despine(right=True, bottom = True)


# %%
#BDNF, Reduction
replace_values_bdnf = {0 : 'Negative', 1 : 'Positive'}
bdnf = data.replace({"BDNF, Reduction": replace_values_bdnf})
my_palette = ["#e74c3c", "#2ecc71"]
sns.set_palette(sns.color_palette(my_palette))
ax = sns.countplot(x="BDNF, Reduction", data=bdnf)
ax.set(ylabel='Number of compounds', xlabel=None, title ='Reduction of BDNF')
sns.despine(right=True, bottom = True)


# %%
#Activity_Syn
replace_values_syn = {0 : 'Unknown', 2: 'Active & Non-selective', 1 : 'Inactive',  3: 'Active & Selective'}
syn = data.replace({"Activity_Syn": replace_values_syn})
my_palette = ["#e74c3c", "salmon", "#2ecc71", "skyblue"]
sns.set_palette(sns.color_palette(my_palette))
ax = sns.countplot(x="Activity_Syn", data=syn, order=[ "Active & Selective", "Active & Non-selective", "Inactive", "Unknown"] )
ax.set(ylabel='Number of compounds', xlabel=None, title ='Synaptogenesis')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
sns.despine(right=True, bottom = True)


# %%
#Activity_NNF
replace_values_nnf = {0 : 'Unknown', 1 : 'Inactive', 2: 'Less potent', 3: 'Potent & Less selective', 4: 'Potent & Selective'}
nnf = data.replace({"Activity_NNF": replace_values_nnf})
my_palette = ["#e74c3c", "salmon", "lightpink", "#2ecc71", "skyblue"]
sns.set_palette(sns.color_palette(my_palette))
ax = sns.countplot(x="Activity_NNF", data=nnf, order = ["Potent & Selective", "Potent & Less selective", "Less potent", "Inactive","Unknown"])
ax.set(ylabel='Number of compounds', xlabel=None, title ='Neural Network Formation')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
sns.despine(right=True, bottom = True)

# %% [markdown]
# ### Distribution of continuous variables

# %%
#SLogP
ax = sns.kdeplot(data['SLogP'], legend = False)
ax.set(xlabel = None, ylabel = None, title ='SLogP')
sns.despine(right=True, bottom = False)


# %%
#Cbrain/Cblood
ax = sns.kdeplot(data['Cbrain/Cblood'], legend=False)
ax.set(xlabel = None, ylabel = None, title ='Cbrain/Cblood')
sns.despine(right=True)


# %%
#Syn_EC30
ax = sns.kdeplot(data['Syn_EC30'], legend=False)
ax.set(ylabel = None, xlabel = 'EC30 (μM)', title ='Synaptogenesis')
sns.despine(right=True)


# %%
#Viability_EC30
ax = sns.kdeplot(data['Viability_EC30'], legend=False)
ax.set(ylabel = None, xlabel = 'EC30 (μM)', title ='Viability Synaptogenesis')
sns.despine(right=True)


# %%
#NNF EC50min
ax = sns.kdeplot(data['NNF EC50min'], legend=False)
ax.set(ylabel = None, xlabel = 'EC50 min (μM)', title ='Neural Network Formation')
sns.despine(right=True)


# %%
#NNF EC50max
ax = sns.kdeplot(data['NNF EC50max'], legend=False)
ax.set(ylabel = None, xlabel = 'EC50 max (μM)', title ='Neural Network Formation')
sns.despine(right=True)


# %%
#Viability_LDH
ax = sns.kdeplot(data['Viability_LDH'], legend=False)
ax.set(ylabel = None, xlabel = 'μM', title ='Viability LDH_NNF')
sns.despine(right=True)

# %% [markdown]
# ### Correlations

# %%
# DNT vs SLogP
replace_values_dnt_slogp = {0 : 'Negative', 1 : 'Positive'}
dnt_slogp = data.replace({"DNT": replace_values_dnt_slogp})
my_palette = ["#e74c3c", "#2ecc71"]
sns.set_palette(sns.color_palette(my_palette))
ax = sns.swarmplot(x=dnt_slogp["DNT"], y=dnt_slogp["SLogP"], data=dnt_slogp)
ax.set(ylabel='SLogP', xlabel=None, title ='DNT vs Lipophilicity')
sns.despine(right=True, bottom = False)


# %%
ax = sns.violinplot(x=dnt_slogp["DNT"], y=dnt_slogp["SLogP"], data=dnt_slogp, inner=None)
ax = sns.swarmplot(x=dnt_slogp["DNT"], y=dnt_slogp["SLogP"], data=dnt_slogp,
                   color="white", edgecolor="gray")
ax.set(ylabel='SLogP', xlabel='DNT Classification', title ='DNT vs Lipophilicity')
sns.despine(right=True, bottom = False)


# %%
# DNT vs Cbrain/Cblood
replace_values_dnt_bbb = {0 : 'Negative', 1 : 'Positive'}
dnt_bbb = data.replace({"DNT": replace_values_dnt_bbb})
my_palette = ["#e74c3c", "#2ecc71"]
sns.set_palette(sns.color_palette(my_palette))
ax = sns.swarmplot(x=dnt_bbb["DNT"], y=dnt_bbb["Cbrain/Cblood"], data=dnt_bbb)
ax.set(ylabel='Cbrain/Cblood', xlabel=None, title ='DNT vs BBB')
sns.despine(right=True, bottom = False)


# %%
ax = sns.violinplot(x=dnt_bbb["DNT"], y=dnt_bbb["Cbrain/Cblood"], data=dnt_bbb, inner=None)
ax = sns.swarmplot(x=dnt_bbb["DNT"], y=dnt_bbb["Cbrain/Cblood"], data=dnt_bbb,
                   color="white", edgecolor="gray")
ax.set(ylabel='Cbrain/Cblood', xlabel='DNT Classification', title ='DNT vs BBB')
sns.despine(right=True, bottom = False)

# %% [markdown]
# ### Missing values

# %%
#Table with percentage
data_new = data.drop(columns=['Chemical', 'CASRN'])
missing = (data_new.isnull().sum(0)/97)*100
df = round(missing)
df

# %% [markdown]
# ### Pairwise relationships

# %%
sns.pairplot(data)

# %% [markdown]
# ### Correlation matrix

# %%
#Non-standardised 
#This one used for manuscript
pearsoncorr = data.corr(method='pearson')
sb.heatmap(pearsoncorr, 
            xticklabels=pearsoncorr.columns,
            yticklabels=pearsoncorr.columns,
            cmap='RdBu_r',
            annot=False,
            linewidth=0.5)


# %%
kendall = data.corr(method='kendall')
sb.heatmap(kendall, 
            xticklabels=pearsoncorr.columns,
            yticklabels=pearsoncorr.columns,
            cmap='RdBu_r',
            annot=False,
            linewidth=0.5)


# %%
spearman = data.corr(method='spearman')
sb.heatmap(spearman, 
            xticklabels=pearsoncorr.columns,
            yticklabels=pearsoncorr.columns,
            cmap='RdBu_r',
            annot=False,
            linewidth=0.5)

# %% [markdown]
# ### Parallel categories diagram

# %%
df = data[['DNT', 'BDNF, Reduction', 'Activity_Syn', 'Activity_NNF']]
fig = px.parallel_categories(df, color_continuous_scale=px.colors.diverging.Tealrose, color_continuous_midpoint=0)

fig.show()


# %%
df = data[['DNT', 'BDNF, Reduction', 'Activity_Syn', 'Activity_NNF']]
fig = px.parallel_coordinates(df, color="Activity_NNF", color_continuous_scale=px.colors.diverging.Tealrose, color_continuous_midpoint=2)
fig.show()


