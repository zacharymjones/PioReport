import streamlit as st
import zipfile
import os
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
import shutil
def extract_zip(zip_bytes):
    try:
        with zipfile.ZipFile(BytesIO(zip_bytes), 'r') as zf:
            extract_dir = os.path.join('extracted', zf.namelist()[0].split('/')[0])
            zf.extractall(path='extracted/')
            return extract_dir, zf.namelist()
    except Exception as e:
        st.error(f'Failed to extract ZIP file: {e}')
        return None, None


def find_report_csv(directory, report_type):
    for root, dirs, files in os.walk(directory):
        if report_type in files:
            return os.path.join(root, report_type)
    return None

def list_extracted_folders(base_dir='extracted'):
    folders = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            folders.append(item)
    return folders

def clean_up_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)


def classify_board_type(cards):
    suits = [card[1] for card in cards.split() if len(card) > 1]
    ranks = [card[0] for card in cards.split() if len(card) > 1]

    unique_suits = set(suits)
    unique_ranks = set(ranks)

    is_paired = any(ranks.count(rank) > 1 for rank in ranks)
    is_monotone = len(unique_suits) == 1

    if is_monotone:
        return 'Monotone boards'
    if is_paired:
        return 'Paired boards'

    rank_values = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10,
                   '9': 9, '8': 8, '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2}

    values = [rank_values[rank] for rank in ranks]
    values.sort(reverse=True)

    if len(values) < 3:
        return 'Unknown'

    high_cards = {'A', 'K', 'Q', 'J', 'T'}
    mid_cards = {'9', '8', '7'}
    low_cards = {'6', '5', '4', '3', '2'}

    if values[0] >= 10 and values[1] >= 10 and values[2] <= 8:
        return 'Two Broadway, one low card'
    if values[0] >= 10 and values[1] >= 10 and values[2] >= 8:
        return 'Two Broadway, connected'
    if values[0] <= 10 and values[1] >= 7 and values[1] <= 10 and values[1] >= 7 and values[2] <= 10:
        return 'Mid-connected'
    if values[0] <= 10 and values[1] <= 7 and values[2] <= 7:
        return 'Low-connected'
    if values[0] == 14 and values[1] <= 9 and values[2] <= 9:
        return 'A-Low connected'
    if values[0] >= 11 and values[1] <= 9 and values[2] <= 9:
        return 'One high card, two mid/low cards'

    return 'Unknown'

def convert_suits_to_symbols(cards):
    suit_mapping = {'s': '♠️', 'd': '♢', 'h': '♡', 'c': '♣️'}
    return ' '.join([card[0] + suit_mapping.get(card[1], card[1]) for card in cards.split()])

def equity_color_scale(value, min_val, max_val, midpoint=50):
    if pd.isna(value):
        return ''
    if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
        return 'background-color: red'
    if value < midpoint:
        relative = (value - min_val) / (midpoint - min_val)
        r, g, b = 255, int(255 * relative), 0
    else:
        relative = (value - midpoint) / (max_val - midpoint)
        r, g, b = int(255 * (1 - relative)), 255, 0
    return f'background-color: rgb({r}, {g}, {b})'


def create_stacked_bar_plot(df):
    # Separate the 'Flop' column and 'CHECK freq' column from the numeric data
    df = df.drop(columns=['OOP Equity', 'IP Equity', 'Board Texture'])
    flop_column = df['Flop']
    check_freq_column = df['CHECK freq']
    df_numeric = df.drop(columns=['Flop', 'CHECK freq'])

    # Reverse the order of the DataFrame
    df_numeric = df_numeric.iloc[::-1]
    flop_column = flop_column.iloc[::-1]
    check_freq_column = check_freq_column.iloc[::-1]

    # Define the colormap
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('', [(0.3,0,0), (0.8,0,0), (1,0.6,0.6), (1,0.9,0.9)])
    # Plot the horizontal bar chart for all columns except 'CHECK freq' with a reversed colormap
    ax = df_numeric.plot.barh(stacked=True, colormap=cmap, figsize = (12,8))
    #plt.rcParams['figure.figsize'] = (12, 8)

    # Add the 'CHECK freq' column to the plot with a green color
    bottom = df_numeric.sum(axis=1).values
    ax.barh(flop_column, check_freq_column, left=bottom, color='green', label='CHECK freq', height=0.5)
    ax.set_title('Strategy', fontsize='x-large')
    ax.set_yticklabels(flop_column, rotation=0, ha='right', fontsize='xx-large')
    ax.set_xlim(0, 100)

    # Adjust the legend position and display the plot
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, df_numeric.columns.tolist() + ['CHECK freq'], bbox_to_anchor=(1, 1), fontsize='x-large')

    st.pyplot(ax.figure)

def stacked_equity_plot(dataframe):
    IP_path = find_report_csv(extract_dir, 'report_IP_Full.csv')
    OOP_path = find_report_csv(extract_dir, 'report_OOP_Full.csv')
    if IP_path and OOP_path:
        IP_df = pd.read_csv(IP_path)
        IP_df = IP_df[IP_df.columns[:4]]
        OOP_df = pd.read_csv(OOP_path)
        OOP_df = OOP_df[OOP_df.columns[:4]]

        player_dfs = [IP_df, OOP_df]

        for df in player_dfs:
            df.columns.values[2] = 'Weight'
            df.columns.values[3] = 'Equity'

            df['Flop'] = df['Flop'].apply(convert_suits_to_symbols)
            df['Board Texture'] = df['Flop'].apply(classify_board_type)
            df = df[df['Board Texture'].isin([selected_texture])]
        unique_flops = IP_df[IP_df['Board Texture'].isin([selected_texture])].Flop.unique()

        buckets_df = pd.DataFrame()
        buckets_df['Flop'] = dataframe['Flop']

        def get_bucket_sums(df, flop):
            flop_df = df[df['Flop'] == flop]
            weak_equity_sum = flop_df[flop_df['Equity'] <= 25]['Weight'].sum()
            okay_equity_sum = flop_df[(flop_df['Equity'] > 25) & (flop_df['Equity'] <= 50)]['Weight'].sum()
            good_equity_sum = flop_df[(flop_df['Equity'] > 50) & (flop_df['Equity'] <= 75)]['Weight'].sum()
            nut_equity_sum = flop_df[flop_df['Equity'] > 75]['Weight'].sum()
            return weak_equity_sum, okay_equity_sum, good_equity_sum, nut_equity_sum

        IP_buckets = []
        OOP_buckets = []

        for flop in report_df['Flop']:
            IP_buckets.append(get_bucket_sums(IP_df, flop))
            OOP_buckets.append(get_bucket_sums(OOP_df, flop))

        IP_buckets_df = pd.DataFrame(IP_buckets, columns=['IP_Weak', 'IP_Okay', 'IP_Good', 'IP_Nut'])
        OOP_buckets_df = pd.DataFrame(OOP_buckets, columns=['OOP_Weak', 'OOP_Okay', 'OOP_Good', 'OOP_Nut'])

        # Convert bucket values to percentages
        def convert_to_percentage(df):
            total = df.sum(axis=1)
            percentage_df = df.divide(total, axis=0) * 100
            return percentage_df

        IP_buckets_df = convert_to_percentage(IP_buckets_df)
        OOP_buckets_df = convert_to_percentage(OOP_buckets_df)

        buckets_df = pd.concat([buckets_df.reset_index(drop=True), IP_buckets_df, OOP_buckets_df], axis=1)


        buckets_df = buckets_df.iloc[::-1]

        gap = 0.15
        # Plotting
        fig, ax = plt.subplots(figsize=(12, 8))
        bar_width = 0.35
        index = range(len(unique_flops))

        # Color map for IP (shades of blue)
        IP_colors = ['#A5D5F3', '#3AA3E4', '#146090', '#072436']

        # Color map for OOP (shades of red)
        #OOP_colors = ['#F3A5A5', '#E43A3A', '#901414', '#360707']
        OOP_colors = ['#A5D5F3', '#3AA3E4', '#146090', '#072436']
        # Plot OOP bars with a gap
        oop_positions = [p + bar_width + gap for p in index]
        ax.barh(oop_positions, buckets_df['OOP_Weak'], bar_width, color=OOP_colors[0])
        ax.barh(oop_positions, buckets_df['OOP_Okay'], bar_width, left=buckets_df['OOP_Weak'], color=OOP_colors[1])
        ax.barh(oop_positions, buckets_df['OOP_Good'], bar_width, left=buckets_df['OOP_Weak'] + buckets_df['OOP_Okay'],
                color=OOP_colors[2])
        ax.barh(oop_positions, buckets_df['OOP_Nut'], bar_width,
                left=buckets_df['OOP_Weak'] + buckets_df['OOP_Okay'] + buckets_df['OOP_Good'], color=OOP_colors[3])

        # Plot IP bars
        ax.barh(index, buckets_df['IP_Weak'], bar_width, color=IP_colors[0], label='Weak Equity <=25%')
        ax.barh(index, buckets_df['IP_Okay'], bar_width, left=buckets_df['IP_Weak'], color=IP_colors[1],
                label='Okay Equity 26-50%')
        ax.barh(index, buckets_df['IP_Good'], bar_width, left=buckets_df['IP_Weak'] + buckets_df['IP_Okay'],
                color=IP_colors[2], label='Good Equity 51-75%')
        ax.barh(index, buckets_df['IP_Nut'], bar_width,
                left=buckets_df['IP_Weak'] + buckets_df['IP_Okay'] + buckets_df['IP_Good'], color=IP_colors[3],
                label='Nut Equity >75%')

        ax.set_title('Equity Buckets (OOP/IP)', fontsize='x-large')
        ax.set_yticks([p + (bar_width + gap) / 2 for p in index])
        ax.set_yticklabels(buckets_df['Flop'], fontsize='xx-large')
        ax.set_xlim(0, 100)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1, 1), fontsize='x-large')

        st.pyplot(fig)

    else:
        st.error('report_IP_Full.csv and/or report_OOP_Full.csv not found.')



st.set_page_config(page_title='PioSOLVER Report', layout='wide')
st.sidebar.title('PioSOLVER Report')


# File upload section
uploaded_file = st.file_uploader('Upload a .zip file', type=['zip'])

extract_dir = None
report_path = None

if uploaded_file is None:
    existing_folders = list_extracted_folders()
    selected_folder = st.selectbox('Or select an already uploaded report folder:', ['None'] + existing_folders)
    if selected_folder != 'None':
        extract_dir = os.path.join('extracted', selected_folder)
        report_path = find_report_csv(extract_dir, 'report.csv')
else:
    selected_folder = None

if uploaded_file is not None and uploaded_file.name.endswith('.zip'):
    extract_dir, file_list = extract_zip(uploaded_file.getvalue())
    if extract_dir is not None:
        report_path = find_report_csv(extract_dir, 'report.csv')
        if report_path is None:
            st.error('report.csv not found in the uploaded ZIP file.')
            clean_up_folder(extract_dir)

if report_path:
    try:
        report_df = pd.read_csv(report_path, skiprows=3)
        report_df = report_df.drop(report_df.index[-1])
        if 'Flop' in report_df.columns:
            report_df['Flop'] = report_df['Flop'].apply(convert_suits_to_symbols)
            report_df['Board Texture'] = report_df['Flop'].apply(classify_board_type)

            # Define equity columns
            equity_columns = ['OOP Equity', 'IP Equity']
            # Define action columns
            action_columns = [col for col in report_df.columns if 'freq' in col]

            all_textures = report_df['Board Texture'].unique().tolist()
            is_filter = st.sidebar.checkbox('Filter by texture', value=True)
            if is_filter:
                selected_texture = st.sidebar.selectbox('Filter by Board Texture:', all_textures)
                report_df = report_df[report_df['Board Texture'].isin([selected_texture])]
            sort_by = st.sidebar.selectbox('Sort by:', report_df.columns[1:-1])
            ascend = st.sidebar.radio('Order: ', ['Largest to smallest', 'Smallest to largest'])
            ascend_bool = True if ascend == 'Smallest to largest' else False
            report_df = report_df.sort_values(by=[sort_by], ascending=ascend_bool)
            # Apply the color scale to OOP Equity

            styled_df = report_df.style.map(
                lambda x: equity_color_scale(x, report_df['OOP Equity'].min(),
                                             report_df['OOP Equity'].max()),
                subset=['OOP Equity']
            )

            # Apply the color scale to IP Equity
            styled_df = styled_df.map(
                lambda x: equity_color_scale(x, report_df['IP Equity'].min(), report_df['IP Equity'].max()),
                subset=['IP Equity']
            )

            # Apply the color scale to each action column individually
            for column in action_columns:
                styled_df = styled_df.map(
                    lambda x, col=column: equity_color_scale(x, report_df[col].min(), report_df[col].max()),
                    subset=[column]
                )
            st.table(styled_df)
            if is_filter:
                create_stacked_bar_plot(report_df)
                stacked_equity_plot(report_df)
        else:
            st.error('Flop data not found in the report.csv.')
    except Exception as e:
        st.error('The .zip you uploaded is not compatible.')
        clean_up_folder(extract_dir)
else:
    st.sidebar.write('IMPORTANT: Use these exact settings for aggregated report')
    st.sidebar.image('report_settings.jpg', use_column_width=True)

