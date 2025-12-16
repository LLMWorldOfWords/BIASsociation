

import networkx as nx
import pandas as pd
import numpy as np
import json
from functools import reduce
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from sklearn import preprocessing
import itertools
import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
#pip -q install git+https://github.com/MassimoStel/emoatlas
import spacy
nlp = spacy.load('en_core_web_lg')
from emoatlas import EmoScores
emo = EmoScores(language="english")

##########################################################################

modelColors = {'Humans': '#3269AF',
               'Mistral': '#F19542',
               'Llama3': '#73BC6B',
               'Haiku': '#6A54A6'}

modelColorPalettes = {'Humans': 'Blues',
                      'Mistral': 'Oranges',
                      'Llama3': 'Greens',
                      'Haiku': 'Purples'}

# Define function to load graphs
def edgelist2graph(path, directed = False):
    edge_list = pd.read_csv(path)
    edge_tuples = zip(edge_list['src'], edge_list['tgt'], edge_list['wt'])
    if directed:
        g = nx.DiGraph()
    else:
        g = nx.Graph()
    g.add_weighted_edges_from(edge_tuples)
    return g

# Load graphs
models = ['Humans', 'Mistral', 'Llama3', 'Haiku']
FA_graphs_filt = {}
for model in models:
    path = './data/graphs/edge_lists/FA_' + model + '_edgelist.csv'
    FA_graphs_filt[model] = edgelist2graph(path, directed = False)

# Find node intersection of graphs
all_nodes_full = []
for model, g in FA_graphs_filt.items():
    all_nodes_full.append(list(g.nodes()))
nodeIntFull = list(reduce(lambda x, y: set(x) & set(y), all_nodes_full))

# Select primes
gender_Primes = ['woman', 'man',
             'girl', 'boy',
             'mother', 'father',
             'female', 'male',
             'feminine', 'masculine'
             ]
rel_eth_sex_Primes = ['christian', 'muslim', 'buddhist', 'jewish', 'atheist',
                      'european', 'african', 'asian', 'indigenous', 'hispanic',
                      'straight', 'gay', 'lesbian', 'bisexual']
political_Primes = ['democrat', 'republican']

# Check that all primes are in all the networks
for word in gender_Primes + rel_eth_sex_Primes + political_Primes:
    if word not in nodeIntFull:
        print(word)

# Save primes
pd.DataFrame({'biasPrime':gender_Primes}).to_csv('./data/primes_targets/gender_primes.csv', index = False)
pd.DataFrame({'biasPrime':rel_eth_sex_Primes}).to_csv('./data/primes_targets/rel_eth_sex_primes.csv', index = False)
pd.DataFrame({'biasPrime':political_Primes}).to_csv('./data/primes_targets/political_primes.csv', index = False)

##########################################################################
# Stereotypes analyses: gender

# Load the gender targets
targets = pd.read_csv('./data/primes_targets/gender_targets.csv')
tgts_F = [x for x in list(targets['Female']) if x in nodeIntFull]
tgts_M = [x for x in list(targets['Male'])if x in nodeIntFull]

# Get the prime pairs
genderPrimePairs = [('woman', 'man'),
                    ('female', 'male'),
                    ('mother', 'father'),
                    ('girl', 'boy'),
                    ('feminine', 'masculine')
                    ] 
primes_F = [x[0] for x in genderPrimePairs]
primes_M = [x[1] for x in genderPrimePairs]

# Load results of spreading activation processes
gender_dfs = {}
for model in models:
    gender_dfs[model] = pd.read_csv('./data/AL_matrices/' + model + '_gender.csv')

# Function to format matrices
def formatMat(df):
    cols = list(df.columns)[1:]
    df_new = df.copy().drop(['node'], axis = 1)
    df_new = pd.DataFrame(df_new)
    df_new.columns = cols
    df_new.index = list(df['node'])
    return df_new

# Function to normalize matrices
def normalizeDF(df, Norm = 'l2', normalize_columns = True, normalize_rows = True):
    df_norm = df.copy()
    if normalize_columns:
        df_norm = preprocessing.normalize(df_norm, norm = Norm, axis = 0)
    if normalize_rows:
        df_norm = preprocessing.normalize(df_norm, norm = Norm, axis = 1)
    df_norm = pd.DataFrame(df_norm)
    df_norm.index = df.index
    df_norm.columns = df.columns
    return df_norm

# Function to filter matrices
def matricesGender(df_norm, rows, primes_F, primes_M):
    mat = df_norm[primes_F + primes_M].loc[rows]
    return mat

# Normalize activation level matrices and get rows (targets) and columns (primes) of interest
matGender = {}
for model, g in FA_graphs_filt.items():
    gender_df = gender_dfs[model]
    gender_df = formatMat(gender_df)
    gender_df_norm = normalizeDF(gender_df, Norm = 'l2', normalize_columns = True, normalize_rows = True)
    matGender[model] = gender_df_norm
    matF = matricesGender(gender_df_norm, tgts_F, primes_F, primes_M)
    matM = matricesGender(gender_df_norm, tgts_M, primes_F, primes_M)
    matGender[model] = {'Female-related targets': matF,
                        'Male-related targets': matM}

# Calculate effect sizes from activation level differences
wilcoxonF = {}
wilcoxonM = {}
AL_differences = {}
for model in models:
    testRes ={}
    AL_differences[model] = {}
    for t in [('Female-related targets', 'greater'), ('Male-related targets', 'less')]:
        data1 = matGender[model][t[0]][primes_F].values.flatten()
        data2 = matGender[model][t[0]][primes_M].values.flatten()
        data = [a - b for (a,b) in zip(data1, data2)]
        AL_differences[model][t[0]] = data
        res = stats.wilcoxon(data, alternative = t[1], method = 'approx')
        p = res.pvalue
        z = res.zstatistic
        effect = z / np.sqrt(len(data))
        if t[1] == 'less':
            testRes[t[0]] = {'p':p, 'effect': -effect}
        else:
            testRes[t[0]] = {'p':p, 'effect': effect}
    wilcoxonF[model] = testRes['Female-related targets']
    wilcoxonM[model] = testRes['Male-related targets']

# Plot effect sizes
wilcoxon = {'Female-related targets': wilcoxonF,
            'Male-related targets': wilcoxonM}
data = wilcoxon
titleDict = {'Female-related targets': 'Female stereotypes',
            'Male-related targets': 'Male stereotypes'}
categories = wilcoxon.keys()
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
plt.subplots_adjust(wspace=0.3, right=0.78)  # leave space on right for legend
bar_width = 0.6
for idx, category in enumerate(categories):
    effects = [data[category][model]['effect'] for model in models]
    p_values = [data[category][model]['p'] for model in models]
    colors = [modelColors[model] for model in models]
    x_positions = np.arange(len(models))
    bars = axes[idx].bar(x_positions, effects, width=bar_width, color=colors, edgecolor='black')
    for bar, p in zip(bars, p_values):
        y_val = bar.get_height()
        if p < 0.001:
            marker = "***"
        elif p < 0.01:
            marker = "**"
        elif p < 0.05:
            marker = "*"
        else:
            marker = ""
        axes[idx].text(bar.get_x() + bar.get_width()/2, y_val + 0.01, marker,
                       ha='center', fontsize=12, fontweight='bold')
    axes[idx].axhline(0, color='black', linewidth=1)
    axes[idx].set_xticks(x_positions)
    axes[idx].set_xticklabels(models, fontsize=12)
    axes[idx].set_title(titleDict[category], fontsize=16)
    #axes[idx].set_xlabel('Model', fontsize=12)
    if idx == 0:
        axes[idx].set_ylabel("Effect Size", fontsize=12)
fig.suptitle("Effect sizes for Gender Stereotypes", fontsize=18, weight='bold')
plt.tight_layout()
plt.show()

# Heatmaps of activation level matrices
def heatmapsGender(dfF, dfM, model, modelColorPalettes):
    # Create the subplots with the desired overall figure size.
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    # Add main title and subtitle.
    plt.suptitle(f'{model}', fontsize=24, y=.95, fontweight = "bold")
    #plt.figtext(0.5, 1, 'Activation Levels', 
                #ha='center', fontsize=20, weight='bold')
    ylab = 'Targets'
    xlab = 'Primes'
    # Create a common colorbar axis on the right side of the figure.
    # Adjust the [left, bottom, width, height] values as needed.
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    # Set colorbar settings to adjust its appearance (thinner colorbar).
    cbar_settings = {"aspect": 20}
    # Plot the first heatmap (Female stereotypes) with the shared colorbar.
    sns.heatmap(dfF, annot=False, cmap=modelColorPalettes[model], ax=axes[0], 
                vmin=0, vmax=1, cbar=True, cbar_ax=cbar_ax, cbar_kws=cbar_settings)
    axes[0].set_title('Female stereotypes', fontsize=20)
    axes[0].set_xticklabels(dfF.columns, fontsize=16, rotation=90)
    axes[0].set_yticklabels(dfF.index, fontsize=16, rotation=0)
    axes[0].set_ylabel(ylab, fontsize=18)
    axes[0].set_xlabel(xlab, fontsize=18)
    # Plot the second heatmap (Male stereotypes) without its own colorbar.
    sns.heatmap(dfM, annot=False, cmap=modelColorPalettes[model], ax=axes[1], 
                vmin=0, vmax=1, cbar=False)
    axes[1].set_title('Male stereotypes', fontsize=20)
    axes[1].set_xticklabels(dfM.columns, fontsize=16, rotation=90)
    axes[1].set_yticklabels(dfM.index, fontsize=16, rotation=0)
    axes[1].set_xlabel(xlab, fontsize=18)
    # Adjust spacing between plots; note the right margin to accommodate the colorbar.
    fig.subplots_adjust(left=0.15, right=0.9, wspace=0.4)
    plt.show()

# Activation level heatmaps
for model in models:
    heatmapsGender(matGender[model]['Female-related targets'], matGender[model]['Male-related targets'], model, modelColorPalettes)

# Mindset streams
def mindsetStreams(start_node, end_node, network):
    emo.plot_mindset_stream(
    graph = list(FA_graphs_filt[network].edges()),
    start_node=start_node,
    end_node=end_node,
    title = network,
    figsize=(12, 7),
    custom_font=14,
    #custom_x_offset = 0,
    marginset=0.15
)
mindsetStreams('feminine','compassionate','Humans')

##########################################################################
# Valence analyses: religion, ethnicity, sexual orientation

# Dictionary of primes for each social identity
primes_dict = {'religion': ['christian', 'muslim', 'buddhist', 'jewish', 'atheist'],
                    'ethnicity': ['european', 'african', 'asian', 'indigenous', 'hispanic'],
                    'sexual orientation': ['straight', 'gay', 'lesbian', 'bisexual']}

# Load results of spreading activation processes
rel_eth_sex_dfs = {}
for model in models:
    rel_eth_sex_dfs[model] = pd.read_csv('./data/AL_matrices/' + model + '_rel_eth_sex.csv')

def loadTextFile(filename):
    file = open(filename, 'r')
    content = file.readlines()
    output = []
    for line in content:
        line = line.strip()
        if '\t' in line:
            line = line.split('\t')
        output.append(line)
    file.close()
    return output

# Load valence data
VAD = loadTextFile('./data/valence_emotions/NRC-VAD-Lexicon.txt')
VAD_Dict = {'valence': {}, 'arousal': {}, 'dominance': {}}
for item in VAD:
    VAD_Dict['valence'][item[0]] = item[1]
    VAD_Dict['arousal'][item[0]] = item[2]
    VAD_Dict['dominance'][item[0]] = item[3]

# Dictionary of activation level matrices
rel_eth_sex_mat = {}
for rel_eth_sex, primes in primes_dict.items():
    model_mat = {}
    for model in models:
        rel_eth_sex_df = rel_eth_sex_dfs[model]
        rel_eth_sex_df = rel_eth_sex_df[['node'] + primes]
        rel_eth_sex_df = formatMat(rel_eth_sex_df)
        model_mat[model] = rel_eth_sex_df
    rel_eth_sex_mat[rel_eth_sex] = model_mat

# Normalize activation level matrices
norm_mat_val = {}
for socialID, modelDict in rel_eth_sex_mat.items():
    norm_mat = {}
    for model, mat in modelDict.items():
        socialID_df_norm = normalizeDF(mat, Norm = 'l2',
                                   normalize_columns = True,
                                   normalize_rows = True)
        norm_mat[model] = socialID_df_norm
    norm_mat_val[socialID] = norm_mat

# Add valence, 
socialID_norm_mat_VAL = {}
for socialID, norm_mat_filt in norm_mat_val.items():
    norm_mat_filt_val = {}
    for model, mat in norm_mat_filt.items():
        matNew = mat.copy()
        d = VAD_Dict['valence']
        matNew['valence'] = [float(d[node]) if node in d.keys() else 'NA' for node in list(mat.index)]
        norm_mat_filt_val[model] = matNew
    socialID_norm_mat_VAL[socialID] = norm_mat_filt_val

def GLM(df, predictors):
    results = {}
    df = df[df['valence'] != 'NA']
    # Fit individual GLMs
    for pred in predictors:
        X = sm.add_constant(df[[pred]].astype(float))
        y = pd.to_numeric(df['valence'], errors='coerce')
        model = sm.GLM(y, X, family=sm.families.Binomial()).fit()
        results[pred] = model.params[pred]
    # Fit full model for Wald test
    X_all = sm.add_constant(df[predictors].astype(float))
    y_all = pd.to_numeric(df['valence'], errors='coerce')
    full_model = sm.GLM(y_all, X_all, family=sm.families.Binomial()).fit()
    wald_formula = " = ".join(predictors)
    wald_test = full_model.wald_test(wald_formula, scalar=False)
    effect = round(wald_test.statistic[0][0], 2)
    p = wald_test.pvalue
    return (results, effect, p)

# Fit GLMs
align = False # True to use only nodes in the intersection (the results do not change significantly)
socialID_Res = {} # Univariate GLM coefficients
socialID_Eff = {} # Multivariate GLM effect sizes and p-values
for socialID, modelDict in socialID_norm_mat_VAL.items():
    modelRes = {}
    modelEff = {}
    for model, df in modelDict.items():
        if align: # use only nodes in the intersection (the results do not change significantly)
            commonRows = list(set(nodeIntFull).intersection(set(df.index)))
            df_new = df.loc[commonRows]
        else:
            df_new = df
        res, eff, p = GLM(df_new, primes_dict[socialID])
        modelRes[model] = res
        modelEff[model] = {'p':p, 'effect': eff}
    socialID_Res[socialID] = modelRes
    socialID_Eff[socialID] = modelEff

# Plot coefficeints of univariate GLMs
socialID_types = ['religion', 'ethnicity', 'sexual orientation']
model_order = ['Humans', 'Mistral', 'Llama3', 'Haiku']
x_positions = {model: i for i, model in enumerate(model_order)}
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
plt.subplots_adjust(wspace=0.3)
# Global y-limits
all_coefs = []
for socialID in socialID_types:
    df = pd.DataFrame(socialID_Res[socialID]).melt()
    all_coefs.extend(df['value'].values)
ymin, ymax = min(all_coefs) - 0.5, max(all_coefs) + 0.5
for ax, socialID in zip(axes, socialID_types):
    coef_dict = socialID_Res[socialID]
    df = pd.DataFrame(coef_dict).reset_index().melt(
        id_vars='index', var_name='model', value_name='coefficient'
    ).rename(columns={'index': 'label'})
    for model in model_order:
        base_x = x_positions[model]
        point_x = base_x - 0.3     # shift scatter and vline slightly left
        text_x = base_x - 0.2     # bring text closer to point
        color = modelColors[model]
        sub_df = df[df['model'] == model].sort_values('coefficient')
        # Points
        ax.scatter([point_x]*len(sub_df), sub_df['coefficient'], color=color, s=60)
        # Update current y-limits after scatter to use for vline
        ax.set_ylim(ymin, ymax)
        # Full-height background vertical line
        ax.vlines(x=point_x, ymin=ymin, ymax=ymax, color=color, alpha=0.2, linewidth=3)
        # Overlap-aware labels
        last_y = None
        min_sep = 0.1
        for _, row in sub_df.iterrows():
            y = row['coefficient']
            if last_y is not None and abs(y - last_y) < min_sep:
                y += min_sep
            ax.text(text_x, y, row['label'],
                    va='center', ha='left', fontsize=10, color=color)
            last_y = y
    # Vertical black dividers between model sections
    for i in range(1, len(model_order)):
        ax.axvline(x=i - 0.5, color='black', linewidth=1.2, alpha=0.7)
    ax.set_xticks(list(x_positions.values()))
    ax.set_xticklabels(model_order, fontsize = 12)
    ax.set_xlim(-0.5, len(model_order) - 0.5)  # Equal panel widths
    ax.axhline(0, color='gray', linestyle='--')
    ax.grid(True, axis='y', linestyle=':', alpha=0.5)
    ax.set_title(socialID.title(), fontsize=16)
    if ax == axes[0]:
        ax.set_ylabel("Coefficient Value", fontsize = 12)
fig.suptitle("Valence GLM coefficients", fontsize=18, weight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Plot effect sizes of multivariate GLMs
data = socialID_Eff
categories = list(data.keys())[:3] 
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
plt.subplots_adjust(wspace=0.4, right=0.82)
bar_width = 0.6
for idx, category in enumerate(categories):
    effects = [data[category][model]['effect'] for model in models]
    p_values = [data[category][model]['p'].item() for model in models]  # Extract float from np.array
    colors = [modelColors[model] for model in models]
    x_positions = np.arange(len(models))
    bars = axes[idx].bar(x_positions, effects, width=bar_width, color=colors, edgecolor='black')
    for bar, p in zip(bars, p_values):
        y_val = bar.get_height()
        if p < 0.001:
            marker = "***"
        elif p < 0.01:
            marker = "**"
        elif p < 0.05:
            marker = "*"
        else:
            marker = ""
        axes[idx].text(bar.get_x() + bar.get_width()/2, y_val + max(effects)*0.01, marker,
                       ha='center', fontsize=12, fontweight='bold')
    axes[idx].axhline(0, color='black', linewidth=1)
    axes[idx].set_xticks(x_positions)
    axes[idx].set_xticklabels(models, fontsize=12)
    axes[idx].set_title(category.title(), fontsize=16)
    #axes[idx].set_xlabel('Model', fontsize=12)
    if idx == 0:
        axes[idx].set_ylabel("Effect Size", fontsize=12)
fig.suptitle("Effect sizes for Valence", fontsize=18, weight='bold')
#handles = [plt.Rectangle((0, 0), 1, 1, color=color, edgecolor='black') for color in modelColors.values()]
#fig.legend(handles, models, title='Network', loc='center left', bbox_to_anchor=(1, 0.85), fontsize=11)
plt.tight_layout()
plt.show()

##########################################################################
# Emotions analyses: political party

# Load results of spreading activation processes
political_dfs = {}
for model in models:
    political_dfs[model] = pd.read_csv('./data/AL_matrices/' + model + '_political.csv')
    
def loadData(filename):
    with open(filename, 'r') as json_file:
        json_list = list(json_file)
    output = []
    for json_str in json_list:
        result = json.loads(json_str)
        output.append(result)
    return output

# Load emotions data
emotions = loadData('./data/valence_emotions/emotions.json')
emoDictEmpty = {'fear':[],
           'sadness': [],
           'surprise': [],
            'anger': [],
            'disgust': [],
            'trust': [],
            'joy': [],
            'anticipation': []}
emoDict = emoDictEmpty.copy()
for k, v in emotions[0].items():
    for emo in emoDict.keys():
        if emo in v:
            emoDict[emo].append(k)
emotionsDict = emoDict

# Activation level matrices
pol_model_mat = {}
for model in models:
    political_df = political_dfs[model]
    political_df = formatMat(political_df)
    pol_model_mat[model] = political_df

# Normalize activation level matrices
pol_norm_mat = {}
for model, mat in pol_model_mat.items():
    pol_df_norm = normalizeDF(mat, Norm = 'l1',
                                normalize_columns = True,
                                normalize_rows = True)
    pol_norm_mat[model] = pol_df_norm

# Get only rows with emotions data
pol_mat_emoRows = {}
for model, mat in pol_norm_mat.items():
    pol_mat_emoRows[model] = mat[mat.index.isin(emotions[0].keys())]

# Add emotion data (columns of 0 or 1) for politics
norm_mat_EMO = {}
for model, mat in pol_mat_emoRows.items():
    emoDictEmpty = {'fear':[],
        'sadness': [],
        'surprise': [],
        'anger': [],
        'disgust': [],
        'trust': [],
        'joy': [],
        'anticipation': []}
    emoDict01 = emoDictEmpty.copy()
    for node in list(mat.index):
        for emo, lst in emotionsDict.items():
            if node in lst:
                emoDict01[emo].append(1)
            else:
                emoDict01[emo].append(0)
    emoDF = pd.DataFrame(emoDict01)
    emoDF.index = mat.index
    matNew = pd.concat([mat, emoDF], axis =1)
    norm_mat_EMO[model] = matNew

# Get the vectors for each emotion for politics
model_vec_emo = {}
for model, mat in norm_mat_EMO.items():
    emo_vec = {}
    for emo in list(emoDict.keys()):
        pol_vec = {}
        for prime in political_Primes:
            mat2 = mat.copy()[mat.copy()[emo] != 0]
            pol_vec[prime] = list(mat2[prime] * mat2[emo])
        emo_vec[emo] = pol_vec
    model_vec_emo[model] = emo_vec

def WRT(data, hyp):
    res = stats.wilcoxon(data, alternative = hyp, method = 'approx')
    p = res.pvalue
    z = res.zstatistic
    effect = z / np.sqrt(len(data))
    return p, effect

# Compute effect sizes for politics
modelEffectsDict = {}
for model, emoDict in model_vec_emo.items():
    emoEffectDict = {}
    for emo, polDict in emoDict.items():
        data = [a-b for (a,b) in zip(polDict[political_Primes[0]], polDict[political_Primes[1]])]
        p, effect = WRT(data, 'two-sided')
        if np.mean(data) > 0: 
            effect *= -1
        emoEffectDict[emo] = {'p': p, 'effect': effect}
    modelEffectsDict[model] = emoEffectDict

# Plot effect sizes
effect_data = modelEffectsDict
model_order = ['Humans', 'Mistral', 'Llama3', 'Haiku']
# Flatten the dictionary into a DataFrame
flat = []
for model in effect_data:
    for emotion in effect_data[model]:
        effect = effect_data[model][emotion]['effect']
        pval = effect_data[model][emotion]['p']
        flat.append({'emotion': emotion, 'model': model, 'effect': effect, 'p': pval})
df = pd.DataFrame(flat)
emotion_order = sorted(df['emotion'].unique())
x = np.arange(len(emotion_order))
bar_width = 0.2
fig, ax = plt.subplots(figsize=(12, 6))
ymin, ymax = df['effect'].min() - 0.1, df['effect'].max() + 0.1
xmin, xmax = df['effect'].min() - 0.1, df['effect'].max() + 0.1
# Shaded background
ax.axhspan(ymin, 0, color='#ffe6e6', zorder=0)
ax.axhspan(0, ymax, color='#e6f0ff', zorder=0)
for i, model in enumerate(model_order):
    model_df = df[df['model'] == model].sort_values('emotion')
    offsets = x + (i - len(model_order)/2) * bar_width + bar_width/2
    bars = ax.bar(offsets, model_df['effect'], width=bar_width,
                    color=modelColors[model], label=model, zorder=3,
                    edgecolor='black')
    # Add p-value stars
    for bar, p in zip(bars, model_df['p']):
        height = bar.get_height()
        if p < 0.001:
            stars = '***'
        elif p < 0.01:
            stars = '**'
        elif p < 0.05:
            stars = '*'
        else:
            stars = ''
        if stars:
            offset = 0.03 if height > 0 else -0.05
            va = 'bottom' if height > 0 else 'top'
            ax.text(bar.get_x() + bar.get_width()/2, height + offset,
                    stars, ha='center', va=va, fontsize=11, fontweight='bold')
for i in x[:-1] + 0.5:
    ax.axvline(i, color='gray', linestyle='-', alpha=0.5, zorder=1)
ax.set_xticks(x)
ax.set_xticklabels(emotion_order, rotation=45, ha='right')
ax.axhline(0, color='gray', linestyle='--', zorder=2)
ax.set_ylabel("Effect Size", weight='bold')
ax.set_xlabel("Emotion", weight='bold')
ax.set_title("Effect sizes for Emotions towards Political Parties", weight='bold', fontsize=16)
ax.set_ylim(ymin, ymax)
ax.grid(True, axis='y', linestyle=':', alpha=0.5, zorder=1)
#ax.legend(title="Model")
ax.legend(title="Model", loc='center left', bbox_to_anchor=(1.02, 0.5))
# First legend: models
model_handles = [
    Patch(facecolor=modelColors[model], edgecolor='black', label=model)
    for model in model_order
]
legend1 = ax.legend(handles=model_handles, loc='upper left', bbox_to_anchor=(1.02, 1),
                    borderaxespad=0, title='Network', frameon=True)
ax.add_artist(legend1)
# Second legend: political regions
background_handles = [
    Patch(facecolor='#e6f0ff', label='democrat'),
    Patch(facecolor='#ffe6e6', label='republican')
]
ax.legend(handles=background_handles, loc='lower left', bbox_to_anchor=(1.02, 0),
        borderaxespad=0, title='Political Party', frameon=True)
plt.tight_layout()
plt.show()
