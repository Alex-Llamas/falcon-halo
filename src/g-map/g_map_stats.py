from optparse import OptionParser
import pandas as pd
import numpy as np
import scipy.linalg
import scipy.stats
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import defaultdict
import os
import math
from sentence_transformers import SentenceTransformer, util

# Arguments
parser = OptionParser("usage: %prog [options]")
parser.add_option("", '--input', action='append', help="Input TSV file(s)")
parser.add_option("", "--output_html", default="nature_figure.html", help="Output HTML filename")
parser.add_option("", "--output_tsv", default="gmap_statistics.tsv", help="Output TSV filename")
parser.add_option("", "--top_dist_output", default="gmap_top_distributions.tsv", help="Top distributions Output TSV filename")
parser.add_option("", "--output_count_hist_html", default="gmap_count_hist.html", help="Count histograms HTML output")
parser.add_option("", "--output_count_hist_stats_tsv", default="gmap_count_hist_stats.tsv", help="Count histograms stats TSV output")
parser.add_option("", "--output_prob_hist_html", default="gmap_prob_hist.html", help="Probability histograms HTML output")
parser.add_option("", "--output_count_prob_stats_tsv", default="gmap_prob_hist_stats.tsv", help="Probability histograms stats TSV output")
parser.add_option("", "--clinical",
                  default="/humgen/diabetes/loki/source/PEGS/paper/figures/figure_3/clinical_trials.csv",
                  help="Clinical trials CSV file")
parser.add_option("", "--corr",
                  default="/humgen/diabetes/loki/pipelines/real_data/birds/projects/cross_trait_eu/cross_trait_eu.all.tsv",
                  help="Correlation TSV")
parser.add_option("", "--heritability",
                  default="/humgen/diabetes/loki/data/heritability_files/aggregated_traits_Mixed.tsv",
                  help="Heritability TSV")
parser.add_option("", "--sample_sizes", default="/humgen/diabetes/loki/data/kp_info/trait_sample_sizes.tsv",
                  help="Sample size TSV")
parser.add_option("", "--effector_list",
                  default="/humgen/diabetes/loki/source/PEGS/paper/figures/figure_3/falcon-egl-index.json",
                  help="Effector genes JSON")
parser.add_option("", "--trait_descriptions",
                  default="/humgen/diabetes/loki/data/kp_info/falcon_traits_with_categories.tsv",
                  help="Trait descriptions TSV file")
parser.add_option("", "--sim_threshold", type="float", default=0.5, help="Similarity threshold for NLP matching")

(options, args) = parser.parse_args()
if not options.input: parser.error("At least one --input file is required.")

TOTAL_HUMAN_GENES = 19_427
print(f"[LOG] Initiating run with {len(options.input)} input files...")
print(f"[LOG] Using NLP similarity threshold: {options.sim_threshold}")


# --- Plotting Helper Functions ---
def create_histogram_json(data, title, xaxis_title, margin, fill_color='rgba(76, 114, 176, 0.5)', line_color='#000000',
                          nbins=None):
    hist_data = np.array(data)
    hist_args = dict(x=hist_data, marker_color=fill_color, marker_line_color=line_color, marker_line_width=1.5)
    if nbins: hist_args['nbinsx'] = nbins
    fig = go.Figure(data=[go.Histogram(**hist_args)])
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color='#333'), x=0.0, xanchor='left'),
        xaxis_title=dict(text=xaxis_title, font=dict(size=10)),
        yaxis_title=dict(text='Frequency', font=dict(size=10)),
        bargap=0.05, template='plotly_white', margin=margin, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig.to_json()


def create_pie_of_pie_json(true_count, false_count, margin, title_text):
    fraction_nearest = true_count / (true_count + false_count) if (true_count + false_count) > 0 else 0.789
    rotation_angle = (90 - 360 * (fraction_nearest + (1 - fraction_nearest) / 2)) % 360

    global_labels = ['Nearest gene', 'Other causes']
    global_values = [true_count, false_count]
    global_colors = ['rgba(74, 136, 201, 0.5)', 'rgba(134, 193, 102, 0.5)']
    global_line_colors = ['rgb(74, 136, 201)', 'rgb(134, 193, 102)']

    breakdown_labels = [
        'S2g', 'Correlation', 'Annotations', 'S2g + Correlation',
        'S2g + Annotations', 'Correlation + Annotations',
        'S2g + Correlation + Annotations', 'Competition'
    ]
    breakdown_values = [1925, 1651, 687, 1342, 1669, 1073, 7849, 7063]
    breakdown_colors = [
        'rgba(223, 73, 59, 0.5)', 'rgba(52, 147, 209, 0.5)', 'rgba(228, 123, 43, 0.5)', 'rgba(145, 82, 158, 0.5)',
        'rgba(198, 66, 57, 0.5)', 'rgba(39, 116, 166, 0.5)', 'rgba(78, 52, 122, 0.5)', 'rgba(38, 184, 94, 0.5)'
    ]
    breakdown_line_colors = [
        'rgb(223, 73, 59)', 'rgb(52, 147, 209)', 'rgb(228, 123, 43)', 'rgb(145, 82, 158)',
        'rgb(198, 66, 57)', 'rgb(39, 116, 166)', 'rgb(78, 52, 122)', 'rgb(38, 184, 94)'
    ]

    uniform_pull_global = [0.03, 0.03]
    uniform_pull_breakdown = [0.03] * len(breakdown_values)

    fig = go.Figure()
    fig.add_trace(go.Pie(
        labels=global_labels, values=global_values, domain=dict(x=[0, 0.40]),
        marker=dict(colors=global_colors, line=dict(color=global_line_colors, width=1.5)),
        textinfo='label+percent', textposition='inside', insidetextorientation='horizontal',
        pull=uniform_pull_global, sort=False, direction='clockwise', rotation=rotation_angle, showlegend=False
    ))
    fig.add_trace(go.Pie(
        labels=breakdown_labels, values=breakdown_values, domain=dict(x=[0.56, 0.96]),
        marker=dict(colors=breakdown_colors, line=dict(color=breakdown_line_colors, width=1.5)),
        textinfo='label', textposition='outside', sort=False, direction='clockwise',
        rotation=20, pull=uniform_pull_breakdown, showlegend=False
    ))
    fig.add_trace(go.Pie(
        labels=breakdown_labels, values=breakdown_values, domain=dict(x=[0.56, 0.96]),
        marker=dict(colors=breakdown_colors, line=dict(color=breakdown_line_colors, width=1.5)),
        textinfo='percent', textposition='inside', sort=False, direction='clockwise',
        rotation=20, pull=uniform_pull_breakdown, showlegend=False
    ))

    fig.update_layout(
        title=dict(text=title_text, font=dict(size=14, color='#333'), x=0.0, xanchor='left'),
        annotations=[
            dict(text='<b>Global</b>', x=0.20, y=1.1, font_size=14, showarrow=False, xanchor='center', xref='paper',
                 yref='paper'),
            dict(text='<b>Breakdown</b>', x=0.76, y=1.1, font_size=14, showarrow=False, xanchor='center', xref='paper',
                 yref='paper')
        ],
        shapes=[
            dict(type='path',
                 path='M 0.43,0.52 L 0.49,0.52 L 0.49,0.56 L 0.54,0.50 L 0.49,0.44 L 0.49,0.48 L 0.43,0.48 Z',
                 fillcolor='rgba(134, 193, 102, 0.5)', line=dict(color='rgb(134, 193, 102)', width=1.5), xref='paper',
                 yref='paper')
        ],
        margin=margin, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(size=10)
    )
    return fig.to_json()


def create_bar_json(counts_dict, title, xaxis_title, margin, fill_color, line_color):
    novel_fill_color = 'rgba(223, 73, 59, 0.6)'
    novel_line_color = 'rgb(223, 73, 59)'
    fig = make_subplots(rows=2, cols=1, row_heights=[0.2, 0.8], vertical_spacing=0.22)

    no_trial_tot = counts_dict['NO TRIAL']['total']
    no_trial_nov = counts_dict['NO TRIAL']['novel']

    fig.add_trace(
        go.Bar(x=[no_trial_tot], y=['No Trial'], orientation='h', marker_color=fill_color, marker_line_color=line_color,
               marker_line_width=1.5, showlegend=False, width=0.6), row=1, col=1)
    fig.add_trace(go.Bar(x=[no_trial_nov], y=['No Trial'], orientation='h', marker_color=novel_fill_color,
                         marker_line_color=novel_line_color, marker_line_width=1.5,
                         text=[no_trial_nov if no_trial_nov > 0 else ''], textposition='auto', showlegend=False,
                         width=0.6), row=1, col=1)

    labels_v = ['Unknown', 'Preclinical', 'Phase 1', 'Phase 1/2', 'Phase 2', 'Phase 2/3', 'Phase 3', 'Approval']
    keys_v = ['UNKNOWN', 'PRECLINICAL', 'PHASE_1', 'PHASE_1_2', 'PHASE_2', 'PHASE_2_3', 'PHASE_3', 'APPROVAL']
    values_v_tot = [counts_dict.get(k, {'total': 0})['total'] for k in keys_v]
    values_v_nov = [counts_dict.get(k, {'novel': 0})['novel'] for k in keys_v]

    fig.add_trace(
        go.Bar(x=labels_v, y=values_v_tot, orientation='v', marker_color=fill_color, marker_line_color=line_color,
               marker_line_width=1.5, showlegend=False), row=2, col=1)
    fig.add_trace(go.Bar(x=labels_v, y=values_v_nov, orientation='v', marker_color=novel_fill_color,
                         marker_line_color=novel_line_color, marker_line_width=1.5,
                         text=[val if val > 0 else '' for val in values_v_nov], textposition='auto', showlegend=False),
                  row=2, col=1)

    fig.update_layout(barmode='overlay',
                      title=dict(text=title, font=dict(size=14, color='#333'), x=0.0, xanchor='left'), bargap=0.2,
                      template='plotly_white', margin=margin, paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)')
    return fig.to_json()


# --- NLP PubMedBERT ---
print("[LOG] Loading PubMedBERT model for NLP similarity matching...")
nlp_model = SentenceTransformer('NeuML/pubmedbert-base-embeddings')


# --- Helper Functions ---
def get_hist_stats(data, is_count=True):
    if not data or len(data) == 0:
        stats = {'mean': 0.0, 'std': 0.0, 'median': 0.0, 'q1': 0.0, 'q3': 0.0, 'total': 0}
        if is_count:
            stats.update({'plus_1': 0, 'exact_1': 0})
        else:
            stats.update({'alpha': 0.0, 'beta': 0.0, 'loc': 0.0, 'scale': 0.0})
        return stats
    d = np.array(data)
    stats = {
        'mean': float(np.mean(d)),
        'std': float(np.std(d)),
        'median': float(np.median(d)),
        'q1': float(np.percentile(d, 25)),
        'q3': float(np.percentile(d, 75))
    }
    if is_count:
        stats['plus_1'] = int(np.sum(d > 1))
        stats['exact_1'] = int(np.sum(d == 1))
        stats['total'] = int(np.sum(d > 0))
    else:
        stats['total'] = len(d)
        try:
            a, b, loc, scale = scipy.stats.beta.fit(d)
            stats.update({'alpha': float(a), 'beta': float(b), 'loc': float(loc), 'scale': float(scale)})
        except:
            stats.update({'alpha': 0.0, 'beta': 0.0, 'loc': 0.0, 'scale': 0.0})
    return stats


def create_genes_dictionary(filepath, threshold):
    filtered_values = defaultdict(list)
    unfiltered_values = defaultdict(list)
    try:
        df = pd.read_csv(filepath, sep='\t')

        # Ensure all columns are uppercase to catch 'Gene', 'gene', 'Symbol' safely
        df.columns = [str(c).strip().upper() for c in df.columns]

        if 'CLUMP' not in df.columns or 'PROBABILITY' not in df.columns:
            return filtered_values, unfiltered_values

        df = df[df['CLUMP'].astype(str).str.upper() != 'NONE'].dropna(subset=['CLUMP'])
        df['PROBABILITY'] = pd.to_numeric(df['PROBABILITY'], errors='coerce').fillna(0)

        if 'NEAREST_TO_LEAD' not in df.columns: df['NEAREST_TO_LEAD'] = False
        if 'GENE' not in df.columns: df['GENE'] = df.index.astype(str)

        df['NEAREST_TO_LEAD'] = df['NEAREST_TO_LEAD'].astype(str).str.strip().str.upper() == 'TRUE'
        df['CLUMP'] = df['CLUMP'].astype(str).str.split(r'(?<!_)-', regex=True)
        df = df.explode('CLUMP')
        df['CLUMP'] = df['CLUMP'].str.strip()
        df = df[df['CLUMP'] != '']
        valid_clumps = df[df['NEAREST_TO_LEAD'] == True]['CLUMP'].unique()
        df = df[df['CLUMP'].isin(valid_clumps)]

        if df.empty: return filtered_values, unfiltered_values
        df['DATA_TUPLE'] = list(zip(df['PROBABILITY'], df['NEAREST_TO_LEAD'], df['GENE']))

        for clump_id, values in df.reset_index(drop=True).groupby('CLUMP')['DATA_TUPLE'].apply(list).items():
            unfiltered_values[filepath + clump_id] = values
    except Exception as e:
        print(f"[ERROR] Reading file {filepath}: {e}")
    return filtered_values, unfiltered_values


# --- Load Trait Categories and Descriptions ---
trait_desc_map = {}
trait_cat_map = {}
if os.path.exists(options.trait_descriptions):
    try:
        desc_df = pd.read_csv(options.trait_descriptions, sep='\t', header=None)
        for _, row in desc_df.iterrows():
            # row[1]: trait name, row[2]: description, row[4]: category
            if len(row) > 2 and pd.notna(row[1]) and pd.notna(row[2]):
                trait_desc_map[str(row[1]).strip()] = str(row[2]).strip()
            if len(row) > 4 and pd.notna(row[1]) and pd.notna(row[4]):
                trait_cat_map[str(row[1]).strip()] = str(row[4]).strip()
        print(f"[LOG] Loaded {len(trait_desc_map)} descriptions and {len(trait_cat_map)} categories.")
    except Exception as e:
        print(f"[WARNING] Could not parse trait descriptions file: {e}")

# --- Load Metadata & Calculate Weights ---
print("\n[LOG] Calculating metadata imputation and weights...")
def get_trait_name(path):
    parts = path.split('/')
    if len(parts) > 9:
        return parts[9]
    return os.path.basename(path).replace('.tsv', '').replace('.csv', '')

name_to_idx = {get_trait_name(path): i for i, path in enumerate(options.input)}
total_traits = len(options.input)
trait_names_list = list(name_to_idx.keys())

h2_df = pd.read_csv(options.heritability, sep='\t') if os.path.exists(options.heritability) else pd.DataFrame(
    columns=['phenotype', 'h2', 'pValue'])
n_df = pd.read_csv(options.sample_sizes, sep='\t') if os.path.exists(options.sample_sizes) else pd.DataFrame(
    columns=['trait', 'sample_size'])

meta_df = pd.DataFrame({'phenotype': trait_names_list})
meta_df = meta_df.merge(h2_df[['phenotype', 'h2', 'pValue']], on='phenotype', how='left')
meta_df = meta_df.merge(n_df[['trait', 'sample_size']], left_on='phenotype', right_on='trait', how='left')

merged_h2 = meta_df['h2'].notna().sum()
merged_n = meta_df['sample_size'].notna().sum()
print(f"[LOG] Matched {merged_h2} / {total_traits} traits with Heritability data.")
print(f"[LOG] Matched {merged_n} / {total_traits} traits with Sample Size data.")

valid_h2 = meta_df['h2'].dropna()
med_h2 = valid_h2.median() if len(valid_h2) > 0 else 0.1
valid_N = meta_df['sample_size'].dropna()
med_N = valid_N.median() if len(valid_N) > 0 else 10000

meta_df['h2_imp'] = meta_df['h2'].fillna(med_h2)
meta_df['N_imp'] = meta_df['sample_size'].fillna(med_N)
meta_df['power'] = meta_df['N_imp'] * meta_df['h2_imp'].clip(lower=0)
P_max = meta_df['power'].max()
if P_max <= 0: P_max = 1.0

meta_df['W_pow'] = np.log(1 + meta_df['power'].astype(float)) / np.log(1 + float(P_max))

dense_mat = np.eye(total_traits, dtype=np.float64)
if options.corr and os.path.exists(options.corr):
    corr_df = pd.read_csv(options.corr, sep='\t')
    for _, row in corr_df.iterrows():
        p1, p2, corr_val = str(row['phenotype_1']), str(row['phenotype_2']), row['correlation']
        if p1 in name_to_idx and p2 in name_to_idx:
            idx1, idx2 = name_to_idx[p1], name_to_idx[p2]
            dense_mat[idx1, idx2] = dense_mat[idx2, idx1] = corr_val

squared_mat = np.square(dense_mat)
meta_df['W_red'] = 1.0 / np.sum(squared_mat, axis=1)
meta_df['W_final'] = meta_df['W_red'] * meta_df['W_pow']
trait_weights = dict(zip(meta_df['phenotype'], meta_df['W_final']))

eigenvalues = scipy.linalg.eigh(dense_mat, eigvals_only=True)
# Sort eigenvalues in descending order
sorted_eigenvalues = np.sort(eigenvalues)[::-1]
sum_eig = np.sum(sorted_eigenvalues)
meff_thresholds = [0.90, 0.95, 0.995]
meff_results = {}
cumulative_eig = np.cumsum(sorted_eigenvalues)
for c in meff_thresholds:
    # Find minimum Meff such that (sum of first Meff eigenvalues) / (total sum) >= c
    m_eff = np.where(cumulative_eig / sum_eig >= c)[0][0] + 1
    meff_results[c] = m_eff
    print(f"[LOG] Calculated Effective Traits (M_eff) for c={c}: {m_eff}")

M = total_traits
# Using the fractional metric for robustness as suggested in the definitions
n_eff_alt = (np.sum(np.maximum(eigenvalues, 0)) ** 2) / np.sum(np.square(np.maximum(eigenvalues, 0)))
print(f"[LOG] Calculated Effective Traits (N_eff - fractional): {n_eff_alt:.2f}")

# --- Extract Pairwise Clinical and Effector Pair Maps ---
print("\n[LOG] Parsing clinical trials and effector gene lists for pair matching...")


CLINICAL_STAGES = ['UNKNOWN', 'PRECLINICAL', 'PHASE_1', 'PHASE_1_2', 'PHASE_2', 'PHASE_2_3', 'PHASE_3', 'APPROVAL']
STAGE_TO_RANK = {s: i for i, s in enumerate(CLINICAL_STAGES)}

def get_phase_rank(p):
    p_str = str(p).upper().replace(' ', '_')
    if 'APPROV' in p_str or 'LAUNCH' in p_str or '4' in p_str:
        return STAGE_TO_RANK['APPROVAL'], 'APPROVAL'

    if p_str in STAGE_TO_RANK:
        return STAGE_TO_RANK[p_str], p_str

    if 'PHASE_3' in p_str: return STAGE_TO_RANK['PHASE_3'], 'PHASE_3'
    if 'PHASE_2' in p_str: return STAGE_TO_RANK['PHASE_2'], 'PHASE_2'
    if 'PHASE_1' in p_str: return STAGE_TO_RANK['PHASE_1'], 'PHASE_1'
    if 'PRE' in p_str: return STAGE_TO_RANK['PRECLINICAL'], 'PRECLINICAL'

    return STAGE_TO_RANK['UNKNOWN'], 'UNKNOWN'


gene_clin_map = defaultdict(list)
approved_genes = set()
if os.path.exists(options.clinical):
    clin_df = pd.read_csv(options.clinical)
    if 'Gene_Name' in clin_df.columns and 'Indication_Name' in clin_df.columns and 'Phase' in clin_df.columns:
        for _, row in clin_df.iterrows():
            g = str(row['Gene_Name']).strip().upper()
            ind = str(row['Indication_Name']).strip()
            p_rank, p_name = get_phase_rank(row['Phase'])
            if p_name and g != 'NAN' and ind != 'NAN':
                gene_clin_map[g].append((ind, p_rank, p_name))
                if p_rank == 4:
                    approved_genes.add(g)

gene_eff_map = defaultdict(set)
if os.path.exists(options.effector_list):
    with open(options.effector_list, 'r') as f:
        egl_data = json.load(f)
        for key, trait_list in egl_data.get('index', {}).items():
            if key == "_": continue
            for g in str(key).split(';'):
                g_clean = g.strip().upper()
                if g_clean:
                    for item in trait_list:
                        if 'trait' in item:
                            gene_eff_map[g_clean].add(item['trait'])

clin_indications = list(set(ind for inds in gene_clin_map.values() for ind, _, _ in inds))
effector_traits = list(set(t for traits in gene_eff_map.values() for t in traits))

print(f"[LOG] Found {len(gene_clin_map)} clinical genes and {len(gene_eff_map)} known effector genes.")
print(
    f"[LOG] Computing pairwise similarity to {len(clin_indications)} indications and {len(effector_traits)} effector traits...")

trait_to_clin = defaultdict(set)
trait_to_eff = defaultdict(set)

if trait_names_list:
    # USE THE DESCRIPTIONS INSTEAD OF RAW NAMES FOR THE EMBEDDINGS
    trait_queries_list = [trait_desc_map.get(t, t) for t in trait_names_list]
    trait_embeddings = nlp_model.encode(trait_queries_list, convert_to_tensor=True)

    if clin_indications:
        clin_embeddings = nlp_model.encode(clin_indications, convert_to_tensor=True)
        cosine_scores_clin = util.cos_sim(trait_embeddings, clin_embeddings)
        for i, t in enumerate(trait_names_list):
            for j, ind in enumerate(clin_indications):
                if cosine_scores_clin[i][j].item() >= options.sim_threshold:
                    trait_to_clin[t].add(ind)

    if effector_traits:
        eff_embeddings = nlp_model.encode(effector_traits, convert_to_tensor=True)
        cosine_scores_eff = util.cos_sim(trait_embeddings, eff_embeddings)
        for i, t in enumerate(trait_names_list):
            for j, eff in enumerate(effector_traits):
                if cosine_scores_eff[i][j].item() >= options.sim_threshold:
                    trait_to_eff[t].add(eff)

print("[LOG] Pairwise NLP mappings complete.")


def evaluate_pair(gene, trait):
    pair_phase_rank, pair_phase_name = -1, 'NO TRIAL'
    matching_indications = trait_to_clin.get(trait, set())
    if gene in gene_clin_map:
        for ind, p_rank, p_name in gene_clin_map[gene]:
            if ind in matching_indications and p_rank > pair_phase_rank:
                pair_phase_rank, pair_phase_name = p_rank, p_name

    has_effector = False
    matching_effs = trait_to_eff.get(trait, set())
    if gene in gene_eff_map:
        for eff in gene_eff_map[gene]:
            if eff in matching_effs:
                has_effector = True
                break

    is_clinical = (pair_phase_rank != -1)
    is_novel = (not is_clinical) and (not has_effector)
    # Repurposable pair: associated, no clinical trial for the pair, but gene has an approved clinical trial for a trait j' != j
    is_repurposable = (not is_clinical) and (gene in approved_genes)
    return pair_phase_rank, pair_phase_name, is_novel, is_repurposable, has_effector, is_clinical


# --- Threshold & Prior Definitions ---
bayes_factors = {
    "Definitive": {"bf": 3162.3, "op": ">="}, "Overwhelming": {"bf": 1000.0, "op": ">="},
    "Compelling": {"bf": 316.2, "op": ">="}, "Extreme": {"bf": 100.0, "op": ">="},
    "Very_Strong": {"bf": 31.6, "op": ">="}, "Strong": {"bf": 10.0, "op": ">="},
    "Moderate": {"bf": 3.16, "op": ">="}, "Anecdotal": {"bf": 1.0, "op": ">"},
    "No_Evidence": {"bf": 1.0, "op": "<="}
}
priors = [1.0, 5.0]


def calc_threshold(category, prior_percent):
    prior = prior_percent / 100.0
    bf = bayes_factors[category]["bf"]
    num = bf * prior
    return num / ((1 - prior) + (bf * prior))


all_raw_data = {}
print("\n[LOG] Extracting clumps from input files...")
for file_path in options.input:
    if not os.path.exists(file_path): continue
    t_name = get_trait_name(file_path)
    _, unfilt_dict = create_genes_dictionary(file_path, -1.0)
    all_raw_data[t_name] = unfilt_dict

total_loci = sum(len(clumps) for clumps in all_raw_data.values())
total_loci_adj = sum(len(clumps) * trait_weights.get(t, 0) for t, clumps in all_raw_data.items())

# --- Generate TSV Statistics Dictionary and Plot Payloads ---
print("\n[LOG] Calculating threshold statistics and HTML graphs...")
stats_list = []


def add_stat(metric, type_val, corr_val, cat_val, bf_val, prior_val, thresh_val, val):
    stats_list.append({
        "Metric_Name": metric, "Type": type_val, "Correction": corr_val,
        "BayesFactorCat": cat_val, "BayesFactorVal": bf_val, "pr_str": prior_val, "Threshold": thresh_val, "Value": val
    })


add_stat("Total_Traits", "Trait_stats", "Raw", "N/A", "N/A", "N/A", "N/A", total_traits)
add_stat("Total_Loci", "Loci_stats", "Raw", "N/A", "N/A", "N/A", "N/A", total_loci)
add_stat("Total_Genes", "Gene_stats", "Raw", "N/A", "N/A", "N/A", "N/A", TOTAL_HUMAN_GENES)
add_stat("Effective_Traits", "Trait_stats", "Corr_adjusted", "N/A", "N/A", "N/A", "N/A", n_eff_alt)
for c, m_eff in meff_results.items():
    add_stat("Effective_Traits", "Trait_stats", "Corr_adjusted", "N/A", "N/A", "N/A", c, m_eff)

precalculated_html_data = {}
precalculated_count_hist_data = {}
precalculated_prob_hist_data = {}

m_pie = dict(l=20, r=20, t=55, b=20)
m_split = dict(l=40, r=20, t=45, b=35)

top_dist_list = []
count_hist_stats_list = []
prob_hist_stats_list = []

for pr in priors:
    pr_str = str(pr)
    precalculated_html_data[pr_str] = {}
    precalculated_count_hist_data[pr_str] = {}
    precalculated_prob_hist_data[pr_str] = {}

    for cat, cat_info in bayes_factors.items():
        cat_threshold = calc_threshold(cat, pr)
        op = cat_info["op"]
        bf_val_str = f"{op} {cat_info['bf']}"

        # To store probabilities for each type
        # types: All, clinical, clinical on stage s, effector, novel and repurposable
        prob_dist = defaultdict(list)
        # To store (gene, trait) pairs for each type
        type_pairs = defaultdict(list)

        trait_genes = defaultdict(set)
        gene_traits = defaultdict(set)

        filt_true_raw, filt_true_adj = 0, 0
        single_gene_loci_raw, single_gene_loci_adj = 0, 0
        f_adjusted_raw, f_adjusted_adj = [], []

        for t, clumps in all_raw_data.items():
            tw = trait_weights.get(t, 0)
            for c, values_list in clumps.items():
                vals = [v for v in values_list if (v[0] >= cat_threshold if op == ">=" else (
                    v[0] > cat_threshold if op == ">" else v[0] <= cat_threshold))]
                if not vals: continue

                sum_post = sum(v[0] for v in vals)
                val_f = np.clip(sum_post - (len(vals) * 0.05), 0, 25)
                f_adjusted_raw.append(val_f)
                f_adjusted_adj.append(val_f * tw)

                if len(vals) == 1:
                    single_gene_loci_raw += 1
                    single_gene_loci_adj += tw

                best_is_nearest = max(vals, key=lambda x: x[0])[1]
                if best_is_nearest:
                    filt_true_raw += 1
                    filt_true_adj += tw

                for prob, is_nearest, gene in vals:
                    trait_genes[t].add(gene)
                    gene_traits[gene].add(t)

                    p_rank, p_name, is_novel, is_repurposable, has_effector, is_clinical = evaluate_pair(gene, t)

                    pair = (gene, t)
                    # All
                    type_pairs['All'].append(pair)
                    prob_dist['All'].append(prob)
                    # Clinical
                    if is_clinical:
                        type_pairs['Clinical'].append(pair)
                        prob_dist['Clinical'].append(prob)
                    # Clinical on stage s
                    if p_name and p_name != 'NO TRIAL':
                        type_pairs[f'Clinical_{p_name}'].append(pair)
                        prob_dist[f'Clinical_{p_name}'].append(prob)
                    # Effector
                    if has_effector:
                        type_pairs['Effector'].append(pair)
                        prob_dist['Effector'].append(prob)
                    # Novel
                    if is_novel:
                        type_pairs['Novel'].append(pair)
                        prob_dist['Novel'].append(prob)
                    # Repurposable
                    if is_repurposable:
                        type_pairs['Repurposable'].append(pair)
                        prob_dist['Repurposable'].append(prob)

        num_associated_genes = len(gene_traits)

        clin_counts_raw = {k: {'total': 0, 'novel': 0} for k in ['NO TRIAL'] + CLINICAL_STAGES}
        clin_counts_adj = {k: {'total': 0, 'novel': 0} for k in ['NO TRIAL'] + CLINICAL_STAGES}

        # We will use these for the count histograms and averages
        # For each type, we need Genes per Trait and Traits per Gene
        # types: All, Clinical, Clinical_{stage}, Effector, Novel, Repurposable

        type_genes_per_trait = defaultdict(lambda: defaultdict(set))
        type_traits_per_gene = defaultdict(lambda: defaultdict(set))

        for tp, pairs in type_pairs.items():
            for g, t in pairs:
                type_genes_per_trait[tp][t].add(g)
                type_traits_per_gene[tp][g].add(t)

        # For Chart (e) specifically
        for g, t in type_pairs['All']:
            p_rank, p_name, is_novel, is_repurposable, has_effector, is_clinical = evaluate_pair(g, t)
            w = trait_weights.get(t, 0)
            clin_counts_raw[p_name]['total'] += 1
            clin_counts_adj[p_name]['total'] += w
            if is_novel:
                clin_counts_raw[p_name]['novel'] += 1
                clin_counts_adj[p_name]['novel'] += w

        # Pre-calculate counts and probabilities for histograms and stats
        # corrections: Raw, Bias_adjusted
        for corr in ['Raw', 'Bias_adjusted']:
            if corr not in precalculated_count_hist_data[pr_str]:
                precalculated_count_hist_data[pr_str][corr] = {}
                precalculated_prob_hist_data[pr_str][corr] = {}
            if cat not in precalculated_count_hist_data[pr_str][corr]:
                precalculated_count_hist_data[pr_str][corr][cat] = {}
                precalculated_prob_hist_data[pr_str][corr][cat] = {}

            for tp in type_pairs.keys():
                # Counts
                genes_per_trait = [len(type_genes_per_trait[tp].get(t, set())) for t in trait_names_list]
                traits_per_gene = [len(type_traits_per_gene[tp].get(g, set())) for g in (gene_traits if tp == 'All' else type_traits_per_gene[tp].keys())]

                if corr == 'Bias_adjusted':
                    genes_per_trait = [len(type_genes_per_trait[tp].get(t, set())) * trait_weights.get(t, 0) for t in trait_names_list]
                    traits_per_gene = [sum(trait_weights.get(t, 0) for t in type_traits_per_gene[tp].get(g, set())) for g in (gene_traits if tp == 'All' else type_traits_per_gene[tp].keys())]

                # Probabilities (Correction doesn't apply to raw probability values, but we might filter or weight if needed?
                # The doc says "probability distributions ... by type ... for each type and threshold".
                # It doesn't explicitly say to weight probabilities. Usually probability histograms are just of the values.)
                probs = prob_dist[tp]

                # Stats for Counts
                gpt_stats = get_hist_stats(genes_per_trait, is_count=True)
                tpg_stats = get_hist_stats(traits_per_gene, is_count=True)

                for metric, sdict in [('Genes_Per_Trait', gpt_stats), ('Traits_Per_Gene', tpg_stats)]:
                    for s_name, s_val in sdict.items():
                        count_hist_stats_list.append({
                            'BayesFactorCat': cat, 'BayesFactorVal': bf_val_str, 'Prior': pr_str,
                            'Threshold': cat_threshold, 'Correction': corr, 'Type': tp,
                            'Metric': metric, 'Stat': s_name, 'Value': s_val
                        })

                # Stats for Probabilities
                prob_stats = get_hist_stats(probs, is_count=False)
                for s_name, s_val in prob_stats.items():
                    prob_hist_stats_list.append({
                        'BayesFactorCat': cat, 'BayesFactorVal': bf_val_str, 'Prior': pr_str,
                        'Threshold': cat_threshold, 'Correction': corr, 'Type': tp,
                        'Metric': 'Probability', 'Stat': s_name, 'Value': s_val
                    })

                # Plots
                precalculated_count_hist_data[pr_str][corr][cat][tp] = {
                    'genes_per_trait': json.loads(create_histogram_json(genes_per_trait, f"Genes per Trait ({tp})", "Count", m_split)),
                    'traits_per_gene': json.loads(create_histogram_json(traits_per_gene, f"Traits per Gene ({tp})", "Count", m_split)),
                    'stats': {'gpt': gpt_stats, 'tpg': tpg_stats},
                    'threshold': cat_threshold,
                    'bf_cat': cat,
                    'bf_val': bf_val_str
                }
                precalculated_prob_hist_data[pr_str][corr][cat][tp] = {
                    'prob_dist': json.loads(create_histogram_json(probs, f"Probability Distribution ({tp})", "Probability", m_split)),
                    'stats': {'prob': prob_stats},
                    'threshold': cat_threshold,
                    'bf_cat': cat,
                    'bf_val': bf_val_str
                }

        # Top 3 Distributions
        for corr_type in ["Raw", "Bias_adjusted"]:
            for p_type in type_pairs.keys():
                g_counts = defaultdict(float)
                t_counts = defaultdict(float)
                cat_counts = defaultdict(float)
                for g, t in type_pairs[p_type]:
                    w_top = 1.0 if corr_type == "Raw" else trait_weights.get(t, 0)
                    g_counts[g] += w_top
                    t_counts[t] += w_top
                    cat_name = trait_cat_map.get(t, "Unknown")
                    cat_counts[cat_name] += w_top

                top_g = sorted(g_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                top_t = sorted(t_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                top_cat = sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)[:3]

                for rank, (name, count) in enumerate(top_g, 1):
                    top_dist_list.append({"BayesFactorCat": cat, "BayesFactorVal": bf_val_str, "Prior": pr_str, "Threshold": cat_threshold, "Correction": corr_type, "Type": p_type, "Metric": "Gene", "Rank": rank, "Name": name, "Count": count})
                for rank, (name, count) in enumerate(top_t, 1):
                    top_dist_list.append({"BayesFactorCat": cat, "BayesFactorVal": bf_val_str, "Prior": pr_str, "Threshold": cat_threshold, "Correction": corr_type, "Type": p_type, "Metric": "Trait", "Rank": rank, "Name": name, "Count": count})
                for rank, (name, count) in enumerate(top_cat, 1):
                    top_dist_list.append({"BayesFactorCat": cat, "BayesFactorVal": bf_val_str, "Prior": pr_str, "Threshold": cat_threshold, "Correction": corr_type, "Type": p_type, "Metric": "Category", "Rank": rank, "Name": name, "Count": count})

        # Main statistics for output_tsv
        # We need to add stats for ALL types (All, Clinical, Clinical_stage, Effector, Novel, Repurposable)
        for corr in ['Raw', 'Bias_adjusted']:
            for tp in type_pairs.keys():
                pairs_count = len(type_pairs[tp])
                if corr == 'Bias_adjusted':
                    pairs_count = sum(trait_weights.get(t, 0) for g, t in type_pairs[tp])

                prefix = tp
                if tp == 'All':
                    add_stat("Total_Associated_Pairs", "Association_stats", corr, cat, bf_val_str, pr_str, cat_threshold, pairs_count)

                    avg_genes_per_trait = pairs_count / total_traits
                    add_stat("Avg_Genes_Per_Trait", "Trait_stats", corr, cat, bf_val_str, pr_str, cat_threshold, avg_genes_per_trait)

                    genetic_signal = avg_genes_per_trait / TOTAL_HUMAN_GENES
                    add_stat("Genetic_signal", "Trait_stats", corr, cat, bf_val_str, pr_str, cat_threshold, genetic_signal)

                    tpg = [sum(trait_weights.get(t, 0) if corr == 'Bias_adjusted' else 1.0 for t in type_traits_per_gene[tp].get(g, set())) for g in gene_traits]
                    pleiotropic_genes = sum(1 for v in tpg if v >= 2)
                    add_stat("Pleiotropic_Genes", "Gene_stats", corr, cat, bf_val_str, pr_str, cat_threshold, pleiotropic_genes)

                    avg_traits_per_gene = pairs_count / TOTAL_HUMAN_GENES
                    add_stat("Avg_Traits_Per_Gene", "Gene_stats", corr, cat, bf_val_str, pr_str, cat_threshold, avg_traits_per_gene)
                else:
                    # e.g. Clinical_Pairs, Avg_Clinical_Genes_Per_Trait, etc.
                    metric_pairs = f"{tp}_Pairs"
                    if tp == 'Novel': metric_pairs = "Novel_Pairs_Discovered"
                    if tp == 'Repurposable': metric_pairs = "Repurposable_Pairs"

                    add_stat(metric_pairs, "Clinical_n_novelty", corr, cat, bf_val_str, pr_str, cat_threshold, pairs_count)

                    metric_gpt = f"Avg_{tp}_Genes_Per_Trait"
                    add_stat(metric_gpt, "Clinical_n_novelty", corr, cat, bf_val_str, pr_str, cat_threshold, pairs_count / total_traits)

                    metric_tpg = f"Avg_{tp}_Traits_Per_Gene"
                    add_stat(metric_tpg, "Clinical_n_novelty", corr, cat, bf_val_str, pr_str, cat_threshold, pairs_count / TOTAL_HUMAN_GENES)

        # HTML Plot generation payload
        total_assoc_pairs_raw = len(type_pairs['All'])
        total_assoc_pairs_adj = sum(trait_weights.get(t, 0) for g, t in type_pairs['All'])

        G_raw_all = [len(type_genes_per_trait['All'].get(t, set())) for t in trait_names_list]
        G_adj_all = [len(type_genes_per_trait['All'].get(t, set())) * trait_weights.get(t, 0) for t in trait_names_list]
        T_raw_all = [len(type_traits_per_gene['All'].get(g, set())) for g in gene_traits]
        T_adj_all = [sum(trait_weights.get(t, 0) for t in type_traits_per_gene['All'].get(g, set())) for g in gene_traits]

        plots = {
            "plot_a": create_pie_of_pie_json(filt_true_raw, total_loci - filt_true_raw, m_pie,
                                             f"<b>a</b> Evidence driving associations ({total_loci:,} loci)"),
            "plot_a_eff": create_pie_of_pie_json(filt_true_adj, total_loci_adj - filt_true_adj, m_pie,
                                                 f"<b>a</b> Evidence driving associations ({total_loci_adj:.1f} effective loci)"),
            "plot_b": create_histogram_json(T_raw_all, "<b>b</b> Gene pleiotropy", "Traits Associated per Gene",
                                            m_split, fill_color='rgba(134, 193, 102, 0.5)'),
            "plot_b_eff": create_histogram_json(T_adj_all, "<b>b</b> Effective Gene pleiotropy",
                                                "Effective Traits Associated per Gene", m_split,
                                                fill_color='rgba(134, 193, 102, 0.5)'),
            "plot_c": create_histogram_json(G_raw_all, "<b>c</b> Trait polygenicity",
                                            "Genes Associated per Trait", m_split,
                                            fill_color='rgba(74, 136, 201, 0.5)'),
            "plot_c_eff": create_histogram_json(G_adj_all, "<b>c</b> Effective Trait polygenicity",
                                                "Effective Genes Associated per Trait", m_split,
                                                fill_color='rgba(74, 136, 201, 0.5)'),
            "plot_d_left": create_histogram_json(f_adjusted_raw, "<b>d</b> Distribution of expected genes",
                                                 "Expected Genes", m_split, fill_color='rgba(145, 82, 158, 0.5)',
                                                 line_color='rgb(145, 82, 158)'),
            "plot_d_left_eff": create_histogram_json(f_adjusted_adj, "<b>d</b> Effective Distribution of expected genes",
                                                     "Effective Expected Genes", m_split, fill_color='rgba(145, 82, 158, 0.5)',
                                                     line_color='rgb(145, 82, 158)'),
            "plot_d_right": create_bar_json(clin_counts_raw,
                                            f"<b>e</b> Clinical and novelty ({total_assoc_pairs_raw:,} pairs)",
                                            "Phase", m_split, fill_color='rgba(228, 123, 43, 0.5)',
                                            line_color='rgb(228, 123, 43)'),
            "plot_d_right_eff": create_bar_json(clin_counts_adj,
                                                f"<b>e</b> Effective Clinical and novelty ({total_assoc_pairs_adj:.1f} effective pairs)",
                                                "Phase", m_split, fill_color='rgba(228, 123, 43, 0.5)',
                                                line_color='rgb(228, 123, 43)')
        }
        precalculated_html_data[pr_str][cat] = {
            "plots": {k: json.loads(v) for k, v in plots.items()},
            "threshold": cat_threshold,
            "bf_cat": cat,
            "bf_val": bf_val_str
        }

        add_stat("Single_Gene_Loci", "Loci_stats", "Raw", cat, bf_val_str, pr_str, cat_threshold, single_gene_loci_raw)
        add_stat("Single_Gene_Loci", "Loci_stats", "Bias_adjusted", cat, bf_val_str, pr_str, cat_threshold, single_gene_loci_adj)
        add_stat("Top_Nearest_Gene_Loci", "Loci_stats", "Raw", cat, bf_val_str, pr_str, cat_threshold, filt_true_raw)
        add_stat("Top_Nearest_Gene_Loci", "Loci_stats", "Bias_adjusted", cat, bf_val_str, pr_str, cat_threshold, filt_true_adj)
        add_stat("Total_Associated_Genes", "Gene_stats", "Raw", cat, bf_val_str, pr_str, cat_threshold, num_associated_genes)
        # Genes_Single_Trait
        tpg_all_raw = [len(type_traits_per_gene['All'].get(g, set())) for g in gene_traits]
        add_stat("Genes_Single_Trait", "Gene_stats", "Raw", cat, bf_val_str, pr_str, cat_threshold, tpg_all_raw.count(1))

stats_df = pd.DataFrame(stats_list,
                        columns=["Metric_Name", "Type", "Correction", "BayesFactorCat", "BayesFactorVal", "pr_str", "Threshold", "Value"])
stats_df.to_csv(options.output_tsv, sep='\t', index=False)
print(f"[LOG] Statistics TSV saved to {options.output_tsv}")

top_dist_df = pd.DataFrame(top_dist_list)
top_dist_df.to_csv(options.top_dist_output, sep='\t', index=False)
print(f"[LOG] Top distributions TSV saved to {options.top_dist_output}")

count_hist_stats_df = pd.DataFrame(count_hist_stats_list)
count_hist_stats_df.to_csv(options.output_count_hist_stats_tsv, sep='\t', index=False)
print(f"[LOG] Count histogram statistics saved to {options.output_count_hist_stats_tsv}")

prob_hist_stats_df = pd.DataFrame(prob_hist_stats_list)
prob_hist_stats_df.to_csv(options.output_count_prob_stats_tsv, sep='\t', index=False)
print(f"[LOG] Probability histogram statistics saved to {options.output_count_prob_stats_tsv}")

js_data_str = json.dumps(precalculated_html_data)
js_count_hist_data_str = json.dumps(precalculated_count_hist_data)
js_prob_hist_data_str = json.dumps(precalculated_prob_hist_data)

# --- Build HTML Dashboard ---
print("\n[LOG] Generating HTML visualization payload...")
html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Nature Figure</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{ font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; color: #222; line-height: 1.5; margin: 20px; background-color: #f3f4f6; }}
        .header-actions {{ position: sticky; top: 0; z-index: 1000; background-color: #f3f4f6; padding: 15px 0 10px 0; max-width: 680px; margin: 0 auto 20px auto; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #d1d5db; }}
        .download-btn {{ display: inline-flex; align-items: center; padding: 8px 16px; background-color: #fff; border: 1px solid #d1d5db; border-radius: 4px; font-size: 14px; font-weight: 600; cursor: pointer; }}
        .figure-container {{ max-width: 680px; margin: 0 auto; padding: 25px; background-color: #ffffff; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }}
        .figure-title {{ font-size: 18px; font-weight: bold; margin-bottom: 15px; padding-bottom: 10px; border-bottom: 1px solid #e5e7eb; }}
        .panel {{ margin-bottom: 10px; width: 100%; }}
        .panel-split-content {{ display: flex; justify-content: space-between; width: 100%; }}
        .plot-half {{ width: 48%; height: 260px; }}
    </style>
</head>
<body>
    <div class="header-actions">
        <button onclick="downloadPDF()" class="download-btn" id="downloadBtn">Download Figure as PDF</button>
        <div class="control-panel" style="display: flex; align-items: center; gap: 15px;">
            <div id="thresholdDisplay" style="font-size: 14px; font-weight: 600; color: #4b5563;">Threshold: --</div>
            <select id="priorDropdown" onchange="updateDashboard()" style="padding: 6px; border-radius: 4px;">
                <option value="1.0">Prior: 1%</option>
                <option value="5.0" selected>Prior: 5%</option>
            </select>
            <select id="filterDropdown" onchange="updateDashboard()" style="padding: 6px; border-radius: 4px;">
                <option value="Definitive">Definitive, >= 3162.3</option>
                <option value="Overwhelming">Overwhelming, >= 1000.0</option>
                <option value="Compelling">Compelling, >= 316.2</option>
                <option value="Extreme">Extreme, >= 100.0</option>
                <option value="Very_Strong">Very Strong, >= 31.6</option>
                <option value="Strong" selected>Strong, >= 10.0</option>
                <option value="Moderate">Moderate, >= 3.16</option>
                <option value="Anecdotal">Anecdotal, > 1.0</option>
                <option value="No_Evidence">No Evidence, <= 1.0</option>
            </select>
            <select id="countModeDropdown" onchange="updateDashboard()" style="padding: 6px; border-radius: 4px;">
                <option value="raw" selected>Counts: Raw</option>
                <option value="effective">Counts: Effective</option>
            </select>
        </div>
    </div>

    <div class="figure-container" id="figure-container">
        <div class="figure-title">Figure 3: G2F-Map Architecture and Polygenicity</div>
        <div class="panel"><div id="plot_a" style="width: 100%; height: 320px;"></div></div>
        <div class="panel">
            <div class="panel-split-content">
                <div id="plot_b" class="plot-half"></div>
                <div id="plot_c" class="plot-half"></div>
            </div>
        </div>
        <div class="panel">
            <div class="panel-split-content">
                <div id="plot_d_left" class="plot-half"></div>
                <div id="plot_d_right" class="plot-half"></div>
            </div>
        </div>
    </div>

    <script>
        const precalcData = {js_data_str};

        function updateDashboard() {{
            let priorVal = document.getElementById('priorDropdown').value; 
            let category = document.getElementById('filterDropdown').value;
            let countMode = document.getElementById('countModeDropdown').value;

            let currentData = precalcData[priorVal][category];
            if (!currentData) return;

            if (currentData.threshold !== undefined) {{
                document.getElementById('thresholdDisplay').innerText = "Threshold: " + currentData.threshold.toFixed(4) + " (" + currentData.bf_cat + ", " + currentData.bf_val + ")";
            }}

            let plots = currentData.plots;

            let p_a = countMode === 'effective' ? plots['plot_a_eff'] : plots['plot_a'];
            if (p_a) Plotly.react('plot_a', p_a.data, p_a.layout, {{displayModeBar: false}});

            let p_b = countMode === 'effective' ? plots['plot_b_eff'] : plots['plot_b'];
            if (p_b) Plotly.react('plot_b', p_b.data, p_b.layout, {{displayModeBar: false}});

            let p_c = countMode === 'effective' ? plots['plot_c_eff'] : plots['plot_c'];
            if (p_c) Plotly.react('plot_c', p_c.data, p_c.layout, {{displayModeBar: false}});

            let p_d_l = countMode === 'effective' ? plots['plot_d_left_eff'] : plots['plot_d_left'];
            if (p_d_l) Plotly.react('plot_d_left', p_d_l.data, p_d_l.layout, {{displayModeBar: false}});

            let p_d_r = countMode === 'effective' ? plots['plot_d_right_eff'] : plots['plot_d_right'];
            if (p_d_r) Plotly.react('plot_d_right', p_d_r.data, p_d_r.layout, {{displayModeBar: false}});
        }}

        updateDashboard();

        function downloadPDF() {{
            const element = document.getElementById('figure-container');
            const btnText = document.getElementById('downloadBtn');
            const originalText = btnText.innerText;
            btnText.innerText = "Generating PDF...";
            html2pdf().set({{
                margin: 10, filename: 'Nature_Figure.pdf', image: {{ type: 'jpeg', quality: 1.0 }},
                html2canvas: {{ scale: 2, useCORS: true, scrollY: 0 }},
                jsPDF: {{ unit: 'mm', format: 'a4', orientation: 'portrait' }},
                pagebreak: {{ mode: 'css' }}
            }}).from(element).save().then(() => btnText.innerText = originalText);
        }}
    </script>
</body>
</html>
"""

with open(options.output_html, "w", encoding='utf-8') as f:
    f.write(html_template)
print(f"[LOG] Figure HTML saved to: {options.output_html}")

def generate_hist_html(data_json, title, output_file, is_prob=False):
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{ font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; margin: 20px; background-color: #f3f4f6; }}
        .controls {{ position: sticky; top: 0; z-index: 1000; background-color: #f3f4f6; padding: 15px; border-bottom: 1px solid #d1d5db; display: flex; gap: 15px; align-items: center; }}
        .container {{ max-width: 1000px; margin: 20px auto; background: white; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        .plot-container {{ display: flex; flex-wrap: wrap; gap: 20px; justify-content: space-around; }}
        .plot {{ width: 48%; height: 400px; }}
        .plot-full {{ width: 100%; height: 500px; }}
    </style>
</head>
<body>
    <div class="controls">
        <div id="thresholdDisplay" style="font-weight: bold;">Threshold: --</div>
        <select id="priorDropdown" onchange="updateTypes(); updatePlots()">
            <option value="1.0">Prior: 1%</option>
            <option value="5.0" selected>Prior: 5%</option>
        </select>
        <select id="bfDropdown" onchange="updateTypes(); updatePlots()">
            <option value="Definitive">Definitive, >= 3162.3</option>
            <option value="Overwhelming">Overwhelming, >= 1000.0</option>
            <option value="Compelling">Compelling, >= 316.2</option>
            <option value="Extreme">Extreme, >= 100.0</option>
            <option value="Very_Strong">Very Strong, >= 31.6</option>
            <option value="Strong" selected>Strong, >= 10.0</option>
            <option value="Moderate">Moderate, >= 3.16</option>
            <option value="Anecdotal">Anecdotal, > 1.0</option>
            <option value="No_Evidence">No Evidence, <= 1.0</option>
        </select>
        <select id="corrDropdown" onchange="updateTypes(); updatePlots()">
            <option value="Raw" selected>Raw</option>
            <option value="Bias_adjusted">Bias Adjusted</option>
        </select>
        <select id="typeDropdown" onchange="updatePlots()">
        </select>
    </div>
    <div class="container">
        <h1>{title}</h1>
        <div id="stats_display" style="margin-bottom: 20px; padding: 10px; background: #e5e7eb; border-radius: 5px; font-family: monospace;"></div>
        <div id="plots" class="plot-container"></div>
    </div>

    <script>
        const allData = {data_json};
        const mainData = {js_data_str};
        const isProb = {str(is_prob).lower()};

        function updateTypes() {{
            const pr = document.getElementById('priorDropdown').value;
            const bf = document.getElementById('bfDropdown').value;
            const corr = document.getElementById('corrDropdown').value;
            const typeDropdown = document.getElementById('typeDropdown');

            const currentSet = allData[pr][corr][bf];
            const types = Object.keys(currentSet).sort();

            const currentVal = typeDropdown.value;
            typeDropdown.innerHTML = '';
            types.forEach(t => {{
                const opt = document.createElement('option');
                opt.value = t;
                opt.innerText = t;
                if (t === 'All') opt.selected = true;
                typeDropdown.appendChild(opt);
            }});
            if (types.includes(currentVal)) typeDropdown.value = currentVal;
        }}

        function formatStats(stats) {{
            let html = '';
            for (let key in stats) {{
                html += '<b>' + key + ':</b> ' + JSON.stringify(stats[key]) + '<br>';
            }}
            return html;
        }}

        function updatePlots() {{
            const pr = document.getElementById('priorDropdown').value;
            const bf = document.getElementById('bfDropdown').value;
            const corr = document.getElementById('corrDropdown').value;
            const tp = document.getElementById('typeDropdown').value;

            const plotsDiv = document.getElementById('plots');
            const statsDiv = document.getElementById('stats_display');
            plotsDiv.innerHTML = '';
            statsDiv.innerHTML = '';

            const data = allData[pr][corr][bf][tp];
            if (!data) return;

            if (data.threshold !== undefined) {{
                document.getElementById('thresholdDisplay').innerText = "Threshold: " + data.threshold.toFixed(4) + " (" + data.bf_cat + ", " + data.bf_val + ")";
            }}
            if (!data) return;

            statsDiv.innerHTML = formatStats(data.stats);

            if (isProb) {{
                const pDiv = document.createElement('div');
                pDiv.className = 'plot-full';
                pDiv.id = 'prob_plot';
                plotsDiv.appendChild(pDiv);
                Plotly.newPlot('prob_plot', data.prob_dist.data, data.prob_dist.layout);
            }} else {{
                const gDiv = document.createElement('div');
                gDiv.className = 'plot';
                gDiv.id = 'gpt_plot';
                plotsDiv.appendChild(gDiv);
                Plotly.newPlot('gpt_plot', data.genes_per_trait.data, data.genes_per_trait.layout);

                const tDiv = document.createElement('div');
                tDiv.className = 'plot';
                tDiv.id = 'tpg_plot';
                plotsDiv.appendChild(tDiv);
                Plotly.newPlot('tpg_plot', data.traits_per_gene.data, data.traits_per_gene.layout);
            }}
        }}

        updateTypes();
        updatePlots();
    </script>
</body>
</html>
"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)

generate_hist_html(js_count_hist_data_str, "Count Histograms", options.output_count_hist_html, is_prob=False)
print(f"[LOG] Count histograms HTML saved to: {options.output_count_hist_html}")

generate_hist_html(js_prob_hist_data_str, "Probability Histograms", options.output_prob_hist_html, is_prob=True)
print(f"[LOG] Probability histograms HTML saved to: {options.output_prob_hist_html}")
print("[LOG] Script execution finished.")