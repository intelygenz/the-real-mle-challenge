from pathlib import Path
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import re
import src.transformer as tr

    

def plot_num_bathroom_from_text_time_test(df_raw):

    plot_data = []
    for n in range(4,8):
        sample_data = np.resize(df_raw.bathrooms_text, 10**n)
        sample = pd.Series(sample_data)
        t1 = datetime.datetime.now()
        _ = sample.apply(tr.num_bathroom_from_text)
        t2 = datetime.datetime.now()
        
        plot_data.append(
            {
                'n': 10**n,
                'time': (t2-t1).total_seconds(),
                'method': 'apply',

            }
        )

        t1 = datetime.datetime.now()
        _ = list(map(tr.num_bathroom_from_text, sample_data))
        t2 = datetime.datetime.now()
        
        plot_data.append(
            {
                'n': 10**n,
                'time': (t2-t1).total_seconds(),
                'method': 'map',

            }
        )

        t1 = datetime.datetime.now()
        _ = [tr.num_bathroom_from_text(text) for text in sample_data]
        t2 = datetime.datetime.now()
        
        plot_data.append(
            {
                'n': 10**n,
                'time': (t2-t1).total_seconds(),
                'method': 'inline_for_loop',

            }
        )

    return pd.DataFrame(plot_data)


def plot_price_to_test_time_test(df_raw):

    plot_data = []
    for n in range(4,8):
        sample_data = np.resize(df_raw.price, 10**n)
        sample = pd.Series(sample_data)
        t1 = datetime.datetime.now()
        _ = sample.str.extract(r"(\d+).").astype(int)
        t2 = datetime.datetime.now()
        
        plot_data.append(
            {
                'n': 10**n,
                'time': (t2-t1).total_seconds(),
                'method': 'apply',

            }
        )

        compiled_pattern = re.compile(r'\d+')
        t1 = datetime.datetime.now()
        _ = list(map(lambda x: int(tr.apply_regex(x, compiled_pattern)), sample_data))
        t2 = datetime.datetime.now()
        
        plot_data.append(
            {
                'n': 10**n,
                'time': (t2-t1).total_seconds(),
                'method': 'map',

            }
        )

        t1 = datetime.datetime.now()
        _ = [int(tr.apply_regex(text, compiled_pattern)) for text in sample_data]
        t2 = datetime.datetime.now()
        
        plot_data.append(
            {
                'n': 10**n,
                'time': (t2-t1).total_seconds(),
                'method': 'inline_for_loop',

            }
        )

    return pd.DataFrame(plot_data)


def plot_pd_cut_time_test(df_raw):

    plot_data = []
    for n in range(4,8):
        sample_data = np.resize(df_raw.price, 10**n)
        sample = pd.Series(sample_data).str.extract(r"(\d+).").astype(int).to_numpy().flatten()
        t1 = datetime.datetime.now()
        _ = pd.cut(sample, bins=[10, 90, 180, 400, np.inf], labels=[0, 1, 2, 3])
        t2 = datetime.datetime.now()
        
        plot_data.append(
            {
                'n': 10**n,
                'time': (t2-t1).total_seconds(),
                'method': 'pd.cut',

            }
        )

        t1 = datetime.datetime.now()
        _ = tr.array_binding(sample, bins=[10, 90, 180, 400, np.inf], labels=[0, 1, 2, 3])
        t2 = datetime.datetime.now()
        
        plot_data.append(
            {
                'n': 10**n,
                'time': (t2-t1).total_seconds(),
                'method': 'numpy',

            }
        )

    return pd.DataFrame(plot_data)



def plot_category_encoder_time_test(df_raw, cols):

    plot_data = []
    for n in range(2,6):
        sample_data = pd.Series(np.resize(df_raw.amenities, 10**n), name='amenities')
        sample = sample_data.reset_index()
        t1 = datetime.datetime.now()
        _ = tr.preprocess_amenities_column(sample)
        t2 = datetime.datetime.now()
        
        plot_data.append(
            {
                'n': 10**n,
                'time': (t2-t1).total_seconds(),
                'method': 'custom_function',

            }
        )

        t1 = datetime.datetime.now()
        _ = sample_data.apply(lambda x: tr.find_categories(x, cols))

        t2 = datetime.datetime.now()
        
        plot_data.append(
            {
                'n': 10**n,
                'time': (t2-t1).total_seconds(),
                'method': 'pd.apply',

            }
        )

        t1 = datetime.datetime.now()
        _ = [tr.find_categories(x, cols) for x in sample_data]

        t2 = datetime.datetime.now()
        
        plot_data.append(
            {
                'n': 10**n,
                'time': (t2-t1).total_seconds(),
                'method': 'loop',

            }
        )


        t1 = datetime.datetime.now()
        _ = list(map(lambda x: tr.find_categories(x, cols), sample_data))

        t2 = datetime.datetime.now()
        
        plot_data.append(
            {
                'n': 10**n,
                'time': (t2-t1).total_seconds(),
                'method': 'map',

            }
        )

    return pd.DataFrame(plot_data)
