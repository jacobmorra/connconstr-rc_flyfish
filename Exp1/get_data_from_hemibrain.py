import bokeh
import bokeh.palettes
import numpy as np
import pandas as pd
import hvplot.pandas
import holoviews as hv
import seaborn as sb
import matplotlib.pyplot as plt

from holoviews import opts
from neuprint import Client, fetch_synapses, fetch_synapse_connections, fetch_neurons, merge_neuron_properties, NeuronCriteria as NC, SynapseCriteria as SC
from bokeh.plotting import figure, output_file, show
from bokeh.plotting import figure, show, output_notebook
hv.extension('bokeh')

#CREATE CLIENT------------------------------------------------------------------------
c = Client('neuprint.janelia.org', dataset='hemibrain:v1.0.1', token='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6ImphY29iam1vcnJhQGdtYWlsLmNvbSIsImxldmVsIjoibm9hdXRoIiwiaW1hZ2UtdXJsIjoiaHR0cHM6Ly9saDUuZ29vZ2xldXNlcmNvbnRlbnQuY29tLy1HYTBfaTk4dElzTS9BQUFBQUFBQUFBSS9BQUFBQUFBQUFBQS9BQ0hpM3JlRU1IODRDQnlCOEZpRVEySVAyLU91dXYtQVRnL3Bob3RvLmpwZz9zej01MD9zej01MCIsImV4cCI6MTc2MDg4ODYzNn0.WyUr6ZydWGiEfMMDWeGWw8X2DbfmLqvZOt1j-DuCBOU')
print(c.fetch_version())

#ROIs = regions that intersect with the neuron ---------------------------------------
from neuprint import fetch_roi_hierarchy
# Show the ROI hierarchy, with primary ROIs marked with '*'
print(fetch_roi_hierarchy(False, mark_primary=True, format='text'))

#NEURON SEARCH CRITERIA = specify neurons by bodyId, type, neuroncriteria-------------
from neuprint import NeuronCriteria as NC
criteria = NC(rois=['AL(R)','CA(R)','LH(R)'])

#FETCH PROPERTIES-----------------------------------------------------------------------
from neuprint import fetch_neurons
neuron_df, roi_counts_df = fetch_neurons(criteria)
print(neuron_df[['bodyId', 'instance', 'type', 'pre', 'post', 'status', 'cropped', 'size']])
print(neuron_df[['bodyId', 'type', 'pre', 'post', 'size']])
print(roi_counts_df)

#FETCH ADJACENCIES (connections between bodies) ------------------------------------------
# = fetch 2 dataframes (first for neuron properties, second for per-ROI connection strengths)
from neuprint import fetch_adjacencies
#ex1 fetch connections between neurons within one olfactory circuit (AL(R)->CA(R)->LH(R))
neuron_df, conn_df = fetch_adjacencies(NC(rois=['AL(R)','CA(R)','LH(R)'], regex=True), NC(rois=['AL(R)','CA(R)','LH(R)'], regex=True))
print(conn_df.sort_values('weight', ascending=False))

#MERGE NEURON PROPERTIES -> Create matrix-------------------------------------------------
from neuprint import merge_neuron_properties
conn_df = merge_neuron_properties(neuron_df, conn_df, ['type', 'instance'])
print(conn_df.columns)

#IS ROI THE PRESYNAPTIC OR POSTSYNAPTIC REGION? ASSUME PRE FOR NOW AS THIS INDICATED ON TUTORIAL
conn_dfBetter = conn_df.sort_values('weight', ascending=False)[['bodyId_pre','bodyId_post','weight','roi']]

#pick the neuron-neuron connections with the greatest weights (i.e. top 10) in the circuit
conn_dfBest = conn_dfBetter.head(800)
print(conn_dfBest)
conn_dfBest.to_csv('top_neurons.csv', index=False)

#CONNECTION MATRIX -------------------------------------------------------------------------
from neuprint.utils import connection_table_to_matrix
matrix = connection_table_to_matrix(conn_df, 'bodyId', sort_by='type')
print(matrix.iloc[:10, :10])
heatmap = matrix.hvplot.heatmap(height=600, width=700).opts(xrotation=60)
hv.save(heatmap,'heat.png')

heat_map = sb.heatmap(matrix2,annot=True)
sb.set(font_scale=1)
heat_map.set_yticklabels(heat_map.get_yticklabels(), rotation=60)
heat_map.set_xticklabels(heat_map.get_xticklabels(), rotation=60)
plt.title("Heat Map of Weights Between Top 20 Neuron Bodies (Ordered by Weight Values)")
plt.xlabel("Neuron Body Id (Presynaptic)")
plt.ylabel("Neuron Body Id (Postsynaptic)")
plt.gcf().subplots_adjust(bottom=0.15)
plt.show()
