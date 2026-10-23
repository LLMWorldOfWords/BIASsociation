"""Save mindset-stream plots for all four models.

Standalone version of the "Mindset streams" section of BiasEval_analyses.py:
it loads only the free-association graphs and writes one PNG per
(model, start word, end word) to figures/mindset_streams/ instead of
showing the plots on screen.
"""
import os
import subprocess
import sys

# emoatlas orders the words in a stream by iterating over Python sets, whose
# order depends on the interpreter's hash seed. Python reads that seed only at
# start-up, so if it isn't fixed yet, re-run this script once with it fixed.
# This makes the plots identical on every run.
HASH_SEED = "0"
if os.environ.get("PYTHONHASHSEED") != HASH_SEED:
    sys.exit(subprocess.call([sys.executable, os.path.abspath(__file__)] + sys.argv[1:],
                             env={**os.environ, "PYTHONHASHSEED": HASH_SEED}))

import matplotlib
matplotlib.use("Agg")  # draw to files only, no plot windows
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from emoatlas import EmoScores

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "figures", "mindset_streams")

models = ['Humans', 'Mistral', 'Llama3', 'Haiku']
pairs = [('feminine', 'compassionate'),
         ('feminine', 'forceful'),
         ('masculine', 'forceful'),
         ('masculine', 'compassionate')]

# Human streams with many paths need a bigger canvas for labels not to overlap
large_plots = {('Humans', 'feminine', 'forceful'),
               ('Humans', 'masculine', 'compassionate')}

emo = EmoScores(language="english")

# Define function to load graphs
def edgelist2graph(path):
    edge_list = pd.read_csv(path)
    g = nx.Graph()
    g.add_weighted_edges_from(zip(edge_list['src'], edge_list['tgt'], edge_list['wt']))
    return g

def mindsetStream(start_node, end_node, network, title, out_path,
                  title_size=10, label_size=12, figsize=(6, 2.5)):
    """Draw one mindset stream and save it to out_path."""
    # emoatlas hardcodes the title at fontsize=16 and calls plt.show() itself,
    # so pause plt.show, draw the plot untitled, then add our own title.
    show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        emo.plot_mindset_stream(
            graph = list(FA_graphs_filt[network].edges()),
            start_node=start_node,
            end_node=end_node,
            title = '',
            figsize=figsize,
            custom_font=label_size,
            marginset=0.15
        )
    finally:
        plt.show = show
    plt.title(title, fontsize=title_size, fontweight="bold")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close("all")

# Load graphs
FA_graphs_filt = {}
for model in models:
    path = os.path.join(HERE, 'data', 'graphs', 'edge_lists', 'FA_' + model + '_edgelist.csv')
    FA_graphs_filt[model] = edgelist2graph(path)

os.makedirs(OUT_DIR, exist_ok=True)
for model in models:
    for i, (start, end) in enumerate(pairs):
        figsize = (12, 7) if (model, start, end) in large_plots else (6, 2.5)
        out_path = os.path.join(OUT_DIR, f"{model}_{start}_{end}.png")
        mindsetStream(start, end, model, model if i == 0 else '', out_path, figsize=figsize)
        print("saved", os.path.relpath(out_path, HERE))
