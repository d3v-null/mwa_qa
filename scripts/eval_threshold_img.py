import pylab
import seaborn as sns
import pandas as pd
import numpy as np
from mycolorpy import colorlist as mcp
from argparse import ArgumentParser

parser = ArgumentParser(
    description="Evaluating cutoff thresholds for power spectrum (delay-transformed visibilities) metrics")
parser.add_argument('tsvfile', type=str, help='Input csv file')
parser.add_argument('-ws', '--wsfile', dest='wsfile', help='tsv containing metadata, e.g ws_stats', type=str,
                    default=None, required=False)
parser.add_argument('-o', '--outfile', dest='outfile', help='thresholding cutoffs', type=str,
                    default=None, required=False)
parser.add_argument('--per_pointing', action='store_true', default=False,
                    help='If True, will devaluate the tcutoff threshold for the individual pointings. Default is False.')
parser.add_argument('--plot', action='store_true',
                    help='Plots the distribution of the metrics', default=None)
parser.add_argument('--save', action='store_true',
                    help='Saves the plots', default=None)
args = parser.parse_args()


def common_indices(list1, list2):
    indices = []
    for i in range(len(list2)):
        if list2[i] in list1:
            indices.append(i)

    return np.unique(indices)


def read_wsstats(wsfile, obsids):
    df = pd.read_csv(wsfile, sep='\t')
    obs = np.unique(np.array(df['OBS']))
    inds = common_indices(obs, obsids)
    return np.array(df['LST DEG'])[inds], np.array(df['CONF'])[inds]


thresh_index=[
    "V RMS BOX NOSUB",
    "PKS INT RATIO VXXYY NOSUB",
    "PKS INT DIFF XXYY SUB",
    "XX PKS0023_026 INT SUB RATIO",
    "YY PKS0023_026 INT SUB RATIO"
]

def evaluate_threshold(ewp, df):
    # calculating IQR (interquartile range)

    outdata = []
    for i, fl in enumerate(thresh_index):

        if args.per_pointing:
            outdict = {}
            outdict['Metric'] = fl
            for pt in np.unique(ewp):
                inds = np.where(np.array(df['EWP']) == pt)
                data = np.array(df[fl])[inds]
                data = data[~np.isnan(data)]
                if len(data) != 0:
                    mean = np.nanmean(data)
                    std = np.nanstd(data)
                    outdict[pt] = tuple((mean - 3 * std, mean + 3 * std))

        else:
            data = np.array(df[fl])
            if len(data) != 0:
                mean = np.nanmean(data)
                std = np.nanstd(data)
                outdata.append(tuple((mean - 3 * std, mean + 3 * std)))

        if args.per_pointing:
            outdata.append(outdict)

    if args.per_pointing:
        df_out = pd.DataFrame(data=outdata, index=thresh_index)

    else:
        df_out = pd.DataFrame(outdata, index=thresh_index,
                              columns=['Lthresh', 'Uthresh'])

    return df_out
# plotting


import matplotlib.pyplot as plt 
from matplotlib import colormaps, colors
import numpy as np
import warnings

def gen_color(cmap,n,reverse=False):
    '''Generates n distinct color from a given colormap.

    Args:
        cmap(str): The name of the colormap you want to use.
            Refer https://matplotlib.org/stable/tutorials/colors/colormaps.html to choose
            Suggestions:
            For Metallicity in Astrophysics: Use coolwarm, bwr, seismic in reverse
            For distinct objects: Use gnuplot, brg, jet,turbo.

        n(int): Number of colors you want from the cmap you entered.

        reverse(bool): False by default. Set it to True if you want the cmap result to be reversed.

    Returns: 
        colorlist(list): A list with hex values of colors.
    '''
    c_map = colormaps.get_cmap(str(cmap)) # select the desired cmap
    arr=np.linspace(0,1,n) #create a list with numbers from 0 to 1 with n items
    colorlist=list()
    for c in arr:
        rgba=c_map(c) #select the rgba value of the cmap at point c which is a number between 0 to 1
        clr=colors.to_hex(rgba) #convert to hex
        colorlist.append(str(clr)) # create a list of these colors
    
    if reverse==True:
        colorlist.reverse()
    return colorlist

def plot_imgqa(dfm, df_out, ewp, figname=None):
    colors = gen_color(cmap='rainbow', n=len(ewp))
    labels = [r'$V_{rms}\, (unsub)$',
              r'$\frac{S_{V}}{(S_{EW} + S_{NS})}$',
              r'Diff $(S_{EW} , S_{NS})$',
              r'$S_{EW}(\frac{sub}{unsub})$',
              r'$S_{NS}(\frac{sub}{unsub})$'
              ]

    fig = pylab.figure(figsize=(12, 8))
    pylab.subplot(2, 3, 1)
    sns.kdeplot(dfm, x='V RMS BOX NOSUB', hue='EWP',
                palette='rainbow', common_norm=True)
    pylab.xlabel('')
    pylab.ylabel('Density', fontsize=16)
    pylab.title(labels[0], size=17)
    # pylab.xlim(-20, 30)
    if args.per_pointing:
        for i, pt in enumerate(ewp):
            pylab.axvline(x=df_out[pt][0][0], ls='dashed', color=colors[i])
            pylab.axvline(x=df_out[pt][0][1], ls='dashed', color=colors[i])
    else:
        pylab.axvline(x=df_out['Lthresh'][0], ls='dashed', color='black')
        pylab.axvline(x=df_out['Uthresh'][0], ls='dashed', color='black')
    pylab.tick_params(labelsize=14, direction='out', length=3, width=1)

    pylab.subplot(2, 3, 2)
    sns.kdeplot(dfm, x='PKS INT RATIO VXXYY NOSUB',
                hue='EWP', palette='rainbow', common_norm=True)
    pylab.xlabel('')
    pylab.title(labels[1], size=17)
    pylab.ylabel('Density', fontsize=16)
    # pylab.xlim(-0.001, 0.07)
    if args.per_pointing:
        for i, pt in enumerate(ewp):
            pylab.axvline(x=df_out[pt][1][0], ls='dashed', color=colors[i])
            pylab.axvline(x=df_out[pt][1][1], ls='dashed', color=colors[i])
    else:
        pylab.axvline(x=df_out['Lthresh'][1], ls='dashed', color='black')
        pylab.axvline(x=df_out['Uthresh'][1], ls='dashed', color='black')
    pylab.tick_params(labelsize=14, direction='out', length=3, width=1)

    pylab.subplot(2, 3, 3)
    sns.kdeplot(dfm, x='PKS INT DIFF XXYY SUB',
                hue='EWP', palette='rainbow', common_norm=True)
    pylab.xlabel('')
    pylab.ylabel('Density', fontsize=16)
    pylab.title(labels[2], size=17)
    if args.per_pointing:
        for i, pt in enumerate(ewp):
            pylab.axvline(x=df_out[pt][2][0], ls='dashed', color=colors[i])
            pylab.axvline(x=df_out[pt][2][1], ls='dashed', color=colors[i])
    else:
        pylab.axvline(x=df_out['Lthresh'][2], ls='dashed', color='black')
        pylab.axvline(x=df_out['Uthresh'][2], ls='dashed', color='black')
    pylab.tick_params(labelsize=14, direction='out', length=3, width=1)

    pylab.subplot(2, 3, 4)
    sns.kdeplot(dfm, x='XX PKS0023_026 INT SUB RATIO',
                hue='EWP', palette='rainbow', common_norm=True)
    pylab.xlabel('')
    pylab.ylabel('Density', fontsize=16)
    pylab.title(labels[3], size=17)
    if args.per_pointing:
        for i, pt in enumerate(ewp):
            pylab.axvline(x=df_out[pt][3][0], ls='dashed', color=colors[i])
            pylab.axvline(x=df_out[pt][3][1], ls='dashed', color=colors[i])
    else:
        pylab.axvline(x=df_out['Lthresh'][3], ls='dashed', color='black')
        pylab.axvline(x=df_out['Uthresh'][3], ls='dashed', color='black')
    pylab.tick_params(labelsize=14, direction='out', length=3, width=1)

    pylab.subplot(2, 3, 5)
    sns.kdeplot(dfm, x='YY PKS0023_026 INT SUB RATIO',
                hue='EWP', palette='rainbow', common_norm=True)
    pylab.xlabel('')
    pylab.ylabel('Density', fontsize=16)
    pylab.title(labels[3], size=17)
    if args.per_pointing:
        for i, pt in enumerate(ewp):
            pylab.axvline(x=df_out[pt][4][0], ls='dashed', color=colors[i])
            pylab.axvline(x=df_out[pt][4][1], ls='dashed', color=colors[i])
    else:
        pylab.axvline(x=df_out['Lthresh'][4], ls='dashed', color='black')
        pylab.axvline(x=df_out['Uthresh'][4], ls='dashed', color='black')
    pylab.tick_params(labelsize=14, direction='out', length=3, width=1)

    pylab.subplots_adjust(hspace=0.3)

    if args.save:
        pylab.savefig(figname)
    else:
        pylab.show()

def get_sub_mask(df):
    """
    get indices for unsub metrics
    """
    names = np.unique(df['IMG NAME'])
    sub_names = [name for name in names if 'sub' in name.lower()]
    assert len(sub_names) == 1, 'Only one sub image is allowed'
    sub_name = sub_names[0]
    return np.array(df['IMG NAME'] == sub_name)

def get_metrics_df(args):
    df = pd.read_csv(args.tsvfile, sep='\t')
    df.dropna(axis=0, how='all', inplace=True)
    df.dropna(axis=1, how='all', inplace=True)
    
    sub_mask = get_sub_mask(df)
    common_cols = ['V RMS BOX', 'XX PKS0023_026 INT', 'YY PKS0023_026 INT', 'V PKS0023_026 INT']
    df_unsub = df.iloc[~sub_mask][['OBS', 'CONF', 'EWP'] + common_cols]
    df_sub = df.iloc[sub_mask][['OBS'] + common_cols]
    
    dfm = pd.merge(df_unsub, df_sub, on='OBS', suffixes=(' NOSUB', ' SUB'))
    
    for sub in ['NOSUB', 'SUB']:
        if {f"XX PKS0023_026 INT {sub}", f"YY PKS0023_026 INT {sub}"}.issubset(set(dfm.columns)):
            dfm[f'PKS INT DIFF XXYY {sub}'] = np.abs(
                dfm[f'XX PKS0023_026 INT {sub}'] - dfm[f'YY PKS0023_026 INT {sub}'])
            if {f"V PKS0023_026 INT {sub}"}.issubset(set(dfm.columns)):
                dfm[f'PKS INT RATIO VXXYY {sub}'] = dfm[f'V PKS0023_026 INT {sub}'] / (
                    dfm[f'XX PKS0023_026 INT {sub}'] + dfm[f'XX PKS0023_026 INT {sub}'])

        for pol in ['XX', 'YY', 'V']:
            if {f"{pol} PKS0023_026 INT {sub}", f"{pol} PKS0023_026 INT NOSUB"}.issubset(set(dfm.columns)):
                dfm[f'{pol} PKS0023_026 INT {sub} RATIO'] = dfm[f'{pol} PKS0023_026 INT {sub}'] / \
                    dfm[f'{pol} PKS0023_026 INT NOSUB']
    print(dfm)
    return dfm

dfm = get_metrics_df(args)

if args.outfile is None:
    outfile = args.tsvfile.replace('.tsv', '_thresholds.tsv')
else:
    outfile = args.outfile
    
dfm.to_csv(outfile, sep='\t', index=False)

# splitting into configuration
# phase I
df_ph1 = dfm.iloc[np.where(dfm['CONF'] == 'Phase I')]
ewp_ph1 = np.unique(df_ph1['EWP'])
df_out_ph1 = evaluate_threshold(ewp_ph1, df_ph1)

if args.plot:
    figure_name = args.tsvfile.replace('.tsv', '_imgqa_dist_PH1.png')
    plot_imgqa(df_ph1, df_out_ph1, ewp_ph1, figure_name)
# writing to tscv file
if args.per_pointing:
    df_out_ph1.to_csv(outfile.replace(
        '.tsv', '_PH1.tsv'), index=False, sep='\t')
else:
    df_out_ph1.to_csv(outfile.replace('.tsv', '_PH1.tsv'),
                      index=False, sep='\t')
# phase II
df_ph2 = dfm.iloc[np.where(dfm['CONF'] == 'Phase II Compact')]
ewp_ph2 = np.unique(df_ph2['EWP'])
df_out_ph2 = evaluate_threshold(ewp_ph2, df_ph2)

if args.plot:
    figure_name = args.tsvfile.replace('.tsv', '_imgqa_dist_PH2.png')
    plot_imgqa(df_ph2, df_out_ph2, ewp_ph2, figure_name)
# writing to tscv file
if args.per_pointing:
    df_out_ph2.to_csv(outfile.replace(
        '.tsv', '_PH2.tsv'), index=False, sep='\t')
else:
    df_out_ph2.to_csv(outfile.replace('.tsv', '_PH2.tsv'),
                      index=False, sep='\t')


for phase, df_out, ewps in [
        ("ph1", df_out_ph1, ewp_ph1), 
        ("ph2", df_out_ph2, ewp_ph2),
]:
    for ewp in ewps:
        print(f"Thresholds for {phase} {ewp=}")
        thresh = df_out.loc["V RMS BOX NOSUB", ewp]
        print(f"    filter_max_vrms_box_nosub={thresh[1]:6.4f}")
        print(f"    filter_min_vrms_box_nosub={thresh[0]:6.4f}")
        thresh = df_out.loc["PKS INT RATIO VXXYY NOSUB", ewp]
        print(f"    filter_max_pks_int_v_ratio_nosub={thresh[1]:6.4f}")
        print(f"    filter_min_pks_int_v_ratio_nosub={thresh[0]:6.4f}")
        thresh = df_out.loc["PKS INT DIFF XXYY SUB", ewp]
        print(f"    filter_max_pks_int_diff_sub={thresh[1]:6.4f}")
        print(f"    filter_min_pks_int_diff_sub={thresh[0]:6.4f}")
        thresh = df_out.loc["XX PKS0023_026 INT SUB RATIO", ewp]
        print(f"    filter_max_pks_int_sub_ratio_xx={thresh[1]:6.4f}")
        print(f"    filter_min_pks_int_sub_ratio_xx={thresh[0]:6.4f}")
        thresh = df_out.loc["YY PKS0023_026 INT SUB RATIO", ewp]
        print(f"    filter_max_pks_int_sub_ratio_yy={thresh[1]:6.4f}")
        print(f"    filter_min_pks_int_sub_ratio_yy={thresh[0]:6.4f}")
        print("")