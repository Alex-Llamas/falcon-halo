from optparse import OptionParser
import pandas as pd
import numpy as np
import scipy.linalg
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

    labels_v = ['Preclinical', 'Phase 1', 'Phase 2', 'Phase 3', 'Approval']
    keys_v = ['PRECLINICAL', 'PHASE 1', 'PHASE 2', 'PHASE 3', 'APPROVAL']
    values_v_tot = [counts_dict[k]['total'] for k in keys_v]
    values_v_nov = [counts_dict[k]['novel'] for k in keys_v]

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
name_to_idx = {path.split('/')[9]: i for i, path in enumerate(options.input)}
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

meta_df['W_pow'] = np.log(1 + meta_df['power']) / np.log(1 + P_max)

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


def get_phase_rank(p):
    p_str = str(p).upper()
    if 'APPROV' in p_str or 'LAUNCH' in p_str or '4' in p_str: return 4, 'APPROVAL'
    if '3' in p_str: return 3, 'PHASE 3'
    if '2' in p_str: return 2, 'PHASE 2'
    if '1' in p_str: return 1, 'PHASE 1'
    if 'PRE' in p_str: return 0, 'PRECLINICAL'
    return -1, None


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
    t_name = file_path.split('/')[9]
    _, unfilt_dict = create_genes_dictionary(file_path, -1.0)
    all_raw_data[t_name] = unfilt_dict

total_loci = sum(len(clumps) for clumps in all_raw_data.values())

# --- Generate TSV Statistics Dictionary and Plot Payloads ---
print("\n[LOG] Calculating threshold statistics and HTML graphs...")
stats_list = []


def add_stat(metric, type_val, corr_val, cat_val, prior_val, thresh_val, val):
    stats_list.append({
        "Metric_Name": metric, "Type": type_val, "Correction": corr_val,
        "cat": cat_val, "pr_str": prior_val, "Threshold": thresh_val, "Value": val
    })


add_stat("Total_Traits", "Trait_stats", "Raw", "N/A", "N/A", "N/A", total_traits)
add_stat("Total_Loci", "Loci_stats", "Raw", "N/A", "N/A", "N/A", total_loci)
add_stat("Total_Genes", "Gene_stats", "Raw", "N/A", "N/A", "N/A", TOTAL_HUMAN_GENES)
add_stat("Effective_Traits", "Trait_stats", "Corr_adjusted", "N/A", "N/A", "N/A", n_eff_alt)
for c, m_eff in meff_results.items():
    add_stat("Effective_Traits", "Trait_stats", "Corr_adjusted", "N/A", "N/A", c, m_eff)

precalculated_html_data = {}
m_pie = dict(l=20, r=20, t=55, b=20)
m_split = dict(l=40, r=20, t=45, b=35)

top_dist_list = []
for pr in priors:
    pr_str = str(pr)
    precalculated_html_data[pr_str] = {}

    for cat, cat_info in bayes_factors.items():
        cat_threshold = calc_threshold(cat, pr)
        op = cat_info["op"]

        trait_genes = defaultdict(set)
        gene_traits = defaultdict(set)
        filt_true, single_gene_loci = 0, 0
        f_adjusted = []

        for t, clumps in all_raw_data.items():
            for c, values_list in clumps.items():
                vals = [v for v in values_list if (v[0] >= cat_threshold if op == ">=" else (
                    v[0] > cat_threshold if op == ">" else v[0] <= cat_threshold))]
                if not vals: continue

                sum_post = sum(v[0] for v in vals)
                f_adjusted.append(np.clip(sum_post - (len(vals) * 0.05), 0, 25))

                if len(vals) == 1: single_gene_loci += 1
                best_is_nearest = max(vals, key=lambda x: x[0])[1]
                if best_is_nearest: filt_true += 1

                for prob, is_nearest, gene in vals:
                    trait_genes[t].add(gene)
                    gene_traits[gene].add(t)

        num_associated_genes = len(gene_traits)

        clin_counts = {k: {'total': 0, 'novel': 0} for k in
                       ['NO TRIAL', 'PRECLINICAL', 'PHASE 1', 'PHASE 2', 'PHASE 3', 'APPROVAL']}

        # Sets of pairs for top distributions
        pairs_all = []
        pairs_novel = []
        pairs_clinical = []
        pairs_repurposable = []

        # Calculation of Raw and Adjusted Statistics
        # G_raw_j: Genes per trait j
        G_raw = {t: len(trait_genes.get(t, set())) for t in trait_names_list}
        # G_adj_j: Adjusted genes per trait j
        G_adj = {t: sum(1.0 * trait_weights.get(t, 0) for _ in trait_genes.get(t, set())) for t in trait_names_list}

        # T_raw_i: Traits per gene i
        T_raw = {g: len(gene_traits.get(g, set())) for g in gene_traits}
        # T_adj_i: Adjusted traits per gene i
        T_adj = {g: sum(trait_weights.get(t, 0) for t in gene_traits.get(g, set())) for g in gene_traits}

        novel_pairs_raw = 0
        novel_pairs_adj = 0
        novel_genes_per_trait = defaultdict(set)
        novel_traits_per_gene = defaultdict(set)

        repurposable_pairs_raw = 0
        repurposable_pairs_adj = 0
        repurposable_genes_per_trait = defaultdict(set)
        repurposable_traits_per_gene = defaultdict(set)

        for g, traits in gene_traits.items():
            for t in traits:
                p_rank, p_name, is_novel, is_repurposable, has_effector, is_clinical = evaluate_pair(g, t)

                clin_counts[p_name]['total'] += 1
                w = trait_weights.get(t, 0)

                pairs_all.append((g, t))
                if is_novel:
                    clin_counts[p_name]['novel'] += 1
                    novel_pairs_raw += 1
                    novel_pairs_adj += w
                    novel_genes_per_trait[t].add(g)
                    novel_traits_per_gene[g].add(t)
                    pairs_novel.append((g, t))
                if is_repurposable:
                    repurposable_pairs_raw += 1
                    repurposable_pairs_adj += w
                    repurposable_genes_per_trait[t].add(g)
                    repurposable_traits_per_gene[g].add(t)
                    pairs_repurposable.append((g, t))
                if is_clinical:
                    pairs_clinical.append((g, t))

        # NovAG_raw_j, NovAG_adj_j
        NovAG_raw = {t: len(novel_genes_per_trait.get(t, set())) for t in trait_names_list}
        NovAG_adj = {t: NovAG_raw[t] * trait_weights.get(t, 0) for t in trait_names_list}
        # NovAT_raw_i, NovAT_adj_i
        NovAT_raw = {g: len(novel_traits_per_gene.get(g, set())) for g in gene_traits}
        NovAT_adj = {g: sum(trait_weights.get(t, 0) for t in novel_traits_per_gene.get(g, set())) for g in gene_traits}

        # RepAG_raw_j, RepAG_adj_j
        RepAG_raw = {t: len(repurposable_genes_per_trait.get(t, set())) for t in trait_names_list}
        RepAG_adj = {t: RepAG_raw[t] * trait_weights.get(t, 0) for t in trait_names_list}
        # RepAT_raw_i, RepAT_adj_i
        RepAT_raw = {g: len(repurposable_traits_per_gene.get(g, set())) for g in gene_traits}
        RepAT_adj = {g: sum(trait_weights.get(t, 0) for t in repurposable_traits_per_gene.get(g, set())) for g in gene_traits}

        novel_genes_count_raw = len(set(g for traits in novel_genes_per_trait.values() for g in traits))

        total_assoc_pairs_raw = sum(G_raw.values())
        total_assoc_pairs_adj = sum(G_adj.values())

        pleiotropic_genes_raw = sum(1 for v in T_raw.values() if v >= 2)
        pleiotropic_genes_adj = sum(1 for v in T_adj.values() if v >= 2)

        avg_traits_per_gene_raw = total_assoc_pairs_raw / TOTAL_HUMAN_GENES
        avg_traits_per_gene_adj = total_assoc_pairs_adj / TOTAL_HUMAN_GENES

        avg_genes_per_trait_raw = total_assoc_pairs_raw / total_traits
        avg_genes_per_trait_adj = total_assoc_pairs_adj / total_traits

        genetic_signal_raw = avg_genes_per_trait_raw / TOTAL_HUMAN_GENES
        genetic_signal_adj = avg_genes_per_trait_adj / TOTAL_HUMAN_GENES

        avg_novel_genes_per_trait_raw = sum(NovAG_raw.values()) / total_traits
        avg_novel_genes_per_trait_adj = sum(NovAG_adj.values()) / total_traits

        avg_novel_traits_per_gene_raw = sum(NovAT_raw.values()) / TOTAL_HUMAN_GENES
        avg_novel_traits_per_gene_adj = sum(NovAT_adj.values()) / TOTAL_HUMAN_GENES

        avg_repurposable_genes_per_trait_raw = sum(RepAG_raw.values()) / total_traits
        avg_repurposable_genes_per_trait_adj = sum(RepAG_adj.values()) / total_traits

        # Using 'l' in variable name to match the metric name string for consistency
        avg_repurposablel_traits_per_gene_raw = sum(RepAT_raw.values()) / TOTAL_HUMAN_GENES
        avg_repurposablel_traits_per_gene_adj = sum(RepAT_adj.values()) / TOTAL_HUMAN_GENES

        # Top 3 Distributions
        for p_list, p_type in [(pairs_all, "All"), (pairs_novel, "Novel"), (pairs_clinical, "Clinical trials"), (pairs_repurposable, "Repurposable")]:
            g_counts = defaultdict(int)
            t_counts = defaultdict(int)
            cat_counts = defaultdict(int)
            for g, t in p_list:
                g_counts[g] += 1
                t_counts[t] += 1
                cat_name = trait_cat_map.get(t, "Unknown")
                cat_counts[cat_name] += 1

            top_g = sorted(g_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            top_t = sorted(t_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            top_cat = sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)[:3]

            for rank, (name, count) in enumerate(top_g, 1):
                top_dist_list.append({"BayesFactor": cat, "Prior": pr_str, "Type": p_type, "Metric": "Gene", "Rank": rank, "Name": name, "Count": count})
            for rank, (name, count) in enumerate(top_t, 1):
                top_dist_list.append({"BayesFactor": cat, "Prior": pr_str, "Type": p_type, "Metric": "Trait", "Rank": rank, "Name": name, "Count": count})
            for rank, (name, count) in enumerate(top_cat, 1):
                top_dist_list.append({"BayesFactor": cat, "Prior": pr_str, "Type": p_type, "Metric": "Category", "Rank": rank, "Name": name, "Count": count})

        # HTML Plot generation payload
        plots = {
            "plot_a": create_pie_of_pie_json(filt_true, total_loci - filt_true, m_pie,
                                             f"<b>a</b> Evidence driving associations ({total_loci:,} loci)"),
            "plot_b": create_histogram_json(list(T_raw.values()), "<b>b</b> Gene pleiotropy", "Traits Associated per Gene",
                                            m_split, fill_color='rgba(134, 193, 102, 0.5)'),
            "plot_b_eff": create_histogram_json(list(T_adj.values()), "<b>b</b> Effective Gene pleiotropy",
                                                "Effective Traits Associated per Gene", m_split,
                                                fill_color='rgba(134, 193, 102, 0.5)'),
            "plot_c": create_histogram_json(list(G_raw.values()), "<b>c</b> Trait polygenicity",
                                            "Genes Associated per Trait", m_split,
                                            fill_color='rgba(74, 136, 201, 0.5)'),
            "plot_c_eff": create_histogram_json(list(G_adj.values()), "<b>c</b> Effective Trait polygenicity",
                                                "Effective Genes Associated per Trait", m_split,
                                                fill_color='rgba(74, 136, 201, 0.5)'),
            "plot_d_left": create_histogram_json(f_adjusted, "<b>d</b> Distribution of expected genes",
                                                 "Expected Genes", m_split, fill_color='rgba(145, 82, 158, 0.5)',
                                                 line_color='rgb(145, 82, 158)'),
            "plot_d_right": create_bar_json(clin_counts,
                                            f"<b>e</b> Clinical and novelty ({total_assoc_pairs_raw:,} pairs)",
                                            "Phase", m_split, fill_color='rgba(228, 123, 43, 0.5)',
                                            line_color='rgb(228, 123, 43)')
        }
        precalculated_html_data[pr_str][cat] = {"plots": {k: json.loads(v) for k, v in plots.items()}}

        add_stat("Single_Gene_Loci", "Loci_stats", "Raw", cat, pr_str, cat_threshold, single_gene_loci)
        add_stat("Top_Nearest_Gene_Loci", "Loci_stats", "Raw", cat, pr_str, cat_threshold, filt_true)
        add_stat("Total_Associated_Genes", "Gene_stats", "Raw", cat, pr_str, cat_threshold, num_associated_genes)
        add_stat("Genes_Single_Trait", "Gene_stats", "Raw", cat, pr_str, cat_threshold, list(T_raw.values()).count(1))

        # Pleiotropic_Genes
        add_stat("Pleiotropic_Genes", "Gene_stats", "Raw", cat, pr_str, cat_threshold, pleiotropic_genes_raw)
        add_stat("Pleiotropic_Genes", "Gene_stats", "Bias_adjusted", cat, pr_str, cat_threshold, pleiotropic_genes_adj)

        # Avg_Traits_Per_Gene
        add_stat("Avg_Traits_Per_Gene", "Gene_stats", "Raw", cat, pr_str, cat_threshold, avg_traits_per_gene_raw)
        add_stat("Avg_Traits_Per_Gene", "Gene_stats", "Bias_adjusted", cat, pr_str, cat_threshold, avg_traits_per_gene_adj)

        # Total_Associated_Pairs
        add_stat("Total_Associated_Pairs", "Association_stats", "Raw", cat, pr_str, cat_threshold, total_assoc_pairs_raw)
        add_stat("Total_Associated_Pairs", "Association_stats", "Bias_adjusted", cat, pr_str, cat_threshold, total_assoc_pairs_adj)

        # Avg_Genes_Per_Trait
        add_stat("Avg_Genes_Per_Trait", "Trait_stats", "Raw", cat, pr_str, cat_threshold, avg_genes_per_trait_raw)
        add_stat("Avg_Genes_Per_Trait", "Trait_stats", "Bias_adjusted", cat, pr_str, cat_threshold, avg_genes_per_trait_adj)

        # Genetic_signal
        add_stat("Genetic_signal", "Trait_stats", "Raw", cat, pr_str, cat_threshold, genetic_signal_raw)
        add_stat("Genetic_signal", "Trait_stats", "Bias_adjusted", cat, pr_str, cat_threshold, genetic_signal_adj)

        # Novel_Pairs_Discovered
        add_stat("Novel_Pairs_Discovered", "Clinical_n_novelty", "Raw", cat, pr_str, cat_threshold, novel_pairs_raw)
        add_stat("Novel_Pairs_Discovered", "Clinical_n_novelty", "Bias_adjusted", cat, pr_str, cat_threshold, novel_pairs_adj)

        # Novel_Genes_Discovered
        add_stat("Novel_Genes_Discovered", "Gene_stats", "Raw", cat, pr_str, cat_threshold, novel_genes_count_raw)

        # Avg_Novel_Genes_Per_Trait
        add_stat("Avg_Novel_Genes_Per_Trait", "Clinical_n_novelty", "Raw", cat, pr_str, cat_threshold, avg_novel_genes_per_trait_raw)
        add_stat("Avg_Novel_Genes_Per_Trait", "Clinical_n_novelty", "Bias_adjusted", cat, pr_str, cat_threshold, avg_novel_genes_per_trait_adj)

        # Avg_Novel_Traits_Per_Gene
        add_stat("Avg_Novel_Traits_Per_Gene", "Clinical_n_novelty", "Raw", cat, pr_str, cat_threshold, avg_novel_traits_per_gene_raw)
        add_stat("Avg_Novel_Traits_Per_Gene", "Clinical_n_novelty", "Bias_adjusted", cat, pr_str, cat_threshold, avg_novel_traits_per_gene_adj)

        # Repurposable_Pairs
        add_stat("Repurposable_Pairs", "Clinical_n_novelty", "Raw", cat, pr_str, cat_threshold, repurposable_pairs_raw)
        add_stat("Repurposable_Pairs", "Clinical_n_novelty", "Bias_adjusted", cat, pr_str, cat_threshold, repurposable_pairs_adj)

        # Avg_Repurposable_Genes_Per_Trait
        add_stat("Avg_Repurposable_Genes_Per_Trait", "Clinical_n_novelty", "Raw", cat, pr_str, cat_threshold, avg_repurposable_genes_per_trait_raw)
        add_stat("Avg_Repurposable_Genes_Per_Trait", "Clinical_n_novelty", "Bias_adjusted", cat, pr_str, cat_threshold, avg_repurposable_genes_per_trait_adj)

        # Avg_Repurposablel_Traits_Per_Gene
        add_stat("Avg_Repurposablel_Traits_Per_Gene", "Clinical_n_novelty", "Raw", cat, pr_str, cat_threshold, avg_repurposablel_traits_per_gene_raw)
        add_stat("Avg_Repurposablel_Traits_Per_Gene", "Clinical_n_novelty", "Bias_adjusted", cat, pr_str, cat_threshold, avg_repurposablel_traits_per_gene_adj)

stats_df = pd.DataFrame(stats_list,
                        columns=["Metric_Name", "Type", "Correction", "cat", "pr_str", "Threshold", "Value"])
stats_df.to_csv(options.output_tsv, sep='\t', index=False)
print(f"[LOG] Statistics TSV saved to {options.output_tsv}")

top_dist_df = pd.DataFrame(top_dist_list)
top_dist_output = "gmap_top_distributions.tsv"
top_dist_df.to_csv(top_dist_output, sep='\t', index=False)
print(f"[LOG] Top distributions TSV saved to {top_dist_output}")

js_data_str = json.dumps(precalculated_html_data)

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
            <select id="priorDropdown" onchange="updateDashboard()" style="padding: 6px; border-radius: 4px;">
                <option value="1.0">Prior: 1%</option>
                <option value="5.0" selected>Prior: 5%</option>
            </select>
            <select id="filterDropdown" onchange="updateDashboard()" style="padding: 6px; border-radius: 4px;">
                <option value="Definitive">Definitive</option>
                <option value="Overwhelming">Overwhelming</option>
                <option value="Compelling">Compelling</option>
                <option value="Extreme">Extreme</option>
                <option value="Very_Strong">Very Strong</option>
                <option value="Strong" selected>Strong</option>
                <option value="Moderate">Moderate</option>
                <option value="Anecdotal">Anecdotal</option>
                <option value="No_Evidence">No Evidence</option>
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

            let plots = currentData.plots;

            if (plots['plot_a']) Plotly.react('plot_a', plots['plot_a'].data, plots['plot_a'].layout, {{displayModeBar: false}});

            let p_b = countMode === 'effective' ? plots['plot_b_eff'] : plots['plot_b'];
            if (p_b) Plotly.react('plot_b', p_b.data, p_b.layout, {{displayModeBar: false}});

            let p_c = countMode === 'effective' ? plots['plot_c_eff'] : plots['plot_c'];
            if (p_c) Plotly.react('plot_c', p_c.data, p_c.layout, {{displayModeBar: false}});

            if (plots['plot_d_left']) Plotly.react('plot_d_left', plots['plot_d_left'].data, plots['plot_d_left'].layout, {{displayModeBar: false}});
            if (plots['plot_d_right']) Plotly.react('plot_d_right', plots['plot_d_right'].data, plots['plot_d_right'].layout, {{displayModeBar: false}});
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
print("[LOG] Script execution finished.")