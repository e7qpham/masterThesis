import matplotlib.pyplot as plt
import pandas as pd
import pypsa, os

import numpy as np
import cartopy.crs as ccrs
import networkx as nx
import pyomo.environ as pe

import warnings
from shapely.errors import ShapelyDeprecationWarning
import gurobipy

from pypsa.plot import add_legend_circles, add_legend_lines, add_legend_patches
from matplotlib.patches import Circle, Patch

import holoviews as hv
#import hvplot.pandas

warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
plt.rc("figure", figsize=(30, 10))
plt.style.use("bmh")


# Load the network data used
case1=pypsa.Network(r".\elec_s_256_lv1.25__Co2L0p29-6H-T-H-B-I-A-solar+p3-dist1_2030.nc")
case2=pypsa.Network(r".\elec_s_256_lv1.25__Co2L0p29-6H-T-H-B-I-A-solar+p3-dist1_2030.nc")
case3=pypsa.Network(r".\elec_s_256_lv1.25__Co2L0p29-6H-T-H-B-I-A-solar+p3-dist1_2030.nc")


# Plot the renewable mix
source1 = case1.generators.groupby("carrier").p_nom_opt.sum()
df = pd.DataFrame(source1).reset_index()
## Aggregating all solar thermal values into one carrier
solar_thermal = df[df['carrier'].str.contains('solar thermal')]['p_nom_opt'].sum()
## Replacing the individual solar thermal values with the total and updating the carrier name
df = df[~df['carrier'].str.contains('solar thermal')]
df = df.append({'carrier': 'solar thermal', 'p_nom_opt': solar_thermal}, ignore_index=True)
df['p_nom_opt (GW)'] = df['p_nom_opt'] / (1e3)
df.set_index("carrier", inplace=True)

fig, ax = plt.subplots(figsize=(10,6))
cumulative_capacity = 0
color_map = plt.get_cmap('tab20').colors
bar_width = 0.05
# Loop through each renewable source and plot
for idx, (carrier, row) in enumerate(df.iterrows()):
    ax.bar('sources', row['p_nom_opt (GW)'], bottom=cumulative_capacity, color=color_map[idx % len(color_map)], label=carrier, width=bar_width)
    cumulative_capacity += row['p_nom_opt (GW)']

ax.set_ylabel('Optimized Capacity (GW)', fontsize=12)
ax.set_title('Total Optimized Capacities by Energy Sources', fontsize=12)
plt.xticks([]) # Hide x-axis labels
plt.legend(title='Carrier', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# Plot the average hydrogen locational marginal price of three cases in 1x3 subplot grid
fig, axs = plt.subplots(3, 1, figsize=(15, 15), subplot_kw={"projection": ccrs.PlateCarree()})
## Define a common extent for all subplots if needed
common_extent = [-10.67, 31.55, 35.29, 70.09]
vmin = min(mean_h2_prices1.min(), mean_h2_prices2.min(), mean_h2_prices3.min())
vmax = max(mean_h2_prices1.max(), mean_h2_prices2.max(), mean_h2_prices3.max())
## Plot Case 1
sc1 = axs[0].scatter(df_H2_1.x, df_H2_1.y, c=mean_h2_prices1, cmap='viridis', vmin=vmin, vmax=vmax, label='H2 LMP')
axs[0].add_feature(cfeature.BORDERS)
axs[0].add_feature(cfeature.COASTLINE)
axs[0].set_extent(common_extent, crs=ccrs.PlateCarree())
#axs[0].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
axs[0].gridlines()
axs[0].set_title('No UG caverns', fontsize=10)
## Plot Case 3
sc2 = axs[1].scatter(df_H2_3.x, df_H2_3.y, c=mean_h2_prices3, cmap='viridis', vmin=vmin, vmax=vmax, label='H2 LMP')
axs[1].add_feature(cfeature.BORDERS)
axs[1].add_feature(cfeature.COASTLINE)
axs[1].set_extent(common_extent, crs=ccrs.PlateCarree())
#axs[1].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
axs[1].gridlines()
axs[1].set_title('Default UG caverns', fontsize=10)
## Plot Case 2
sc3 = axs[2].scatter(df_H2_2.x, df_H2_2.y, c=mean_h2_prices2, cmap='viridis', vmin=vmin, vmax=vmax, label='H2 LMP')
axs[2].add_feature(cfeature.BORDERS)
axs[2].add_feature(cfeature.COASTLINE)
axs[2].set_extent(common_extent, crs=ccrs.PlateCarree())
#axs[2].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
axs[2].gridlines()
axs[2].set_title('FNB UG caverns', fontsize=10)
## Add a colorbar to the last plot
cbar = plt.colorbar(sc3, ax=axs, orientation='vertical', shrink=0.5, pad=0.1)
cbar.set_label('Average Hydrogen LMP (€/MWh)')
## Set a common label for the y-axis if needed
#fig.text(0.05, 0.5, 'Latitude', va='center', rotation='vertical')
## Set a common label for the x-axis if needed
#fig.text(0.5, 0.04, 'Longitude', ha='center')
plt.show()


# Plot the hydrogen grid in Europe for three cases
def plot_h2_network_subplot(network, ax, link_width_divisor, legend_labels, legend_widths):
    n = network.copy()
    assign_location(n)  # this function assigns x, y locations to the buses
    n.mremove("Link", n.links[~n.links.carrier.str.contains("H2 pipeline")].index)

    n.links.bus0 = n.links.bus0.str.replace(" H2", "")
    n.links.bus1 = n.links.bus1.str.replace(" H2", "")

    for idx in n.links.loc[n.links.bus0 > n.links.bus1].index:
        n.links.at[idx, 'bus0'], n.links.at[idx, 'bus1'] = n.links.at[idx, 'bus1'], n.links.at[idx, 'bus0']

    # Aggregate the capacities if there are parallel links
    n.links.index = n.links.apply(lambda x: f"H2 pipeline {x.bus0} -> {x.bus1}", axis=1)
    n.links = n.links.groupby(n.links.index).agg(
        dict(bus0="first", bus1="first", carrier="first", p_nom_opt="sum")
    )

    # Plot the network
    n.plot(
        bus_sizes=0.02,
        link_colors="green", 
        link_widths=n.links.p_nom_opt / link_width_divisor,
        branch_components=["Link"],
        ax=ax,
        geomap=True
    )

    legend_lines = [mlines.Line2D([], [], color="green", linewidth=width, label=label)
                    for label, width in zip(legend_labels, legend_widths)]
    
    legend = ax.legend(handles=legend_lines, loc='upper left', frameon=False, 
                       title='Hydrogen network [GW]', title_fontsize='medium', 
                       labelspacing=1.5, handletextpad=2, borderpad=1)

    ax.add_artist(legend)
    ax.gridlines()

fig, axes = plt.subplots(1, 3, figsize=(15, 15), subplot_kw={'projection': ccrs.PlateCarree()})
common_extent = [-10.67, 31.55, 35.29, 70.09]
fig.suptitle('Hydrogen grid in Europe')

# Plot each scenario on its respective axis
plot_h2_network_subplot(case1, axes[0], link_width_divisor=4e2, legend_labels=["1 GW", "3 GW"], legend_widths=[1,4])
plot_h2_network_subplot(case3, axes[1], link_width_divisor=5e2, legend_labels=["10 GW", "70 GW"], legend_widths=[2,6])
plot_h2_network_subplot(case2, axes[2], link_width_divisor=5e2, legend_labels=["10 GW", "70 GW"], legend_widths=[2,6])

axes[0].set_title('No UG caverns', fontsize=10)
axes[1].set_title('Default UG caverns', fontsize=10)
axes[2].set_title('FNB UG caverns', fontsize=10)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()



