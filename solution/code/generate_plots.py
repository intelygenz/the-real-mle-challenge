from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from code.src.plots import *

DIR_REPO = Path.cwd().parent
DIR_DATA_RAW = Path(DIR_REPO) / "data" / "raw"
FILEPATH_DATA = DIR_DATA_RAW / "listings.csv"
FILEPATH_PLOTS = Path(DIR_REPO) / "solution" / "plots"
CAT_COLS = ['TV', 'Internet', 'Air conditioning', 'Kitchen', 'Heating', 'Wifi', 'Elevator', 'Breakfast'] 



df_raw = pd.read_csv(FILEPATH_DATA, low_memory=False)

print("Generating bathroom func time test plot")
plot = plot_num_bathroom_from_text_time_test(df_raw)
fig1 = sns.lineplot(plot, x='n', y='time', hue='method').set_title('Time optimizer function: num_bathroom_from_text_time')
plt.savefig(FILEPATH_PLOTS / "bathroom.png")
plt.close()

print("Generating price func time test plot")
plot = plot_price_to_test_time_test(df_raw)
fig2 = sns.lineplot(plot, x='n', y='time', hue='method').set_title('Time optimizer function: price_text')
plt.savefig(FILEPATH_PLOTS / "price.png")
plt.close()

print("Generating pd.cut func time test plot")
plot = plot_pd_cut_time_test(df_raw)
fig3 = sns.lineplot(plot, x='n', y='time', hue='method').set_title('Time optimizer function: price_text')
plt.savefig(FILEPATH_PLOTS / "pd_cut.png")
plt.close()

print("Generating category encoder func time test plot")
plot = plot_category_encoder_time_test(df_raw, CAT_COLS)
fig4 = sns.lineplot(plot, x='n', y='time', hue='method').set_title('Time optimizer function: preprocess_amenities_column')
plt.savefig(FILEPATH_PLOTS / "cat_encoder.png")
plt.close()
