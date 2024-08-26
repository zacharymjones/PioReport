# PioSOLVER Aggregated Report Analytics
https://piosolver2.streamlit.app/

This Streamlit application processes '.zip' files containing PioSOLVER post-flop reports, extracts the relevant data, and provides a rich visualization of the strategy and equity distribution for both In-Position (IP) and Out-of-Position (OOP) players across various board textures. The app allows for deep analysis of PioSOLVER data through interactive filtering and custom visualizations.
## Features

- **Upload & Extract ZIP Reports**: Upload PioSOLVER report ZIP files, which the application extracts and processes.
- **Strategy Analysis**: Displays poker action frequencies and equity metrics in color-coded tables.
- **Board Classification**: Automatically classifies flop boards based on their texture (e.g., monotone, paired, broadway).
- **Stacked Bar Charts**: Visualize the strategy frequencies across multiple board types with stacked horizontal bar charts.
- **Equity Buckets**: Displays the distribution of equity buckets (weak, okay, good, nut equity) for both in-position (IP) and out-of-position (OOP) players.


- **File Upload**: Upload .zip files containing PioSOLVER reports.
- **Folder Management**: Extracts the uploaded .zip files and stores them for later access.
- **Board Classification**: Automatically classifies boards by texture, e.g., Monotone, Paired, etc.
- **Equity Distribution Visualization**: Visualize equity distributions across different flop textures using interactive stacked bar charts.
- **Action Frequency Visualization**: Display stacked bar plots showing the frequencies of actions like checks, bets, and raises for each board texture.
- **Dynamic Filtering**: Filter reports by board texture and equity categories.
- **Sorting and Highlighting**: Sort data by equity, action frequency, and apply conditional formatting to emphasize values.
## Usage

- **Uploading a Report**: Upload a .zip file containing the PioSOLVER reports. The app will automatically extract the data and display relevant information.
- **Filtering:** Use the sidebar to filter the data by board texture, and sort by various columns such as OOP Equity, IP Equity, or specific action frequencies like CHECK freq.
- **Folder Management**: If you have already uploaded reports, select them from the dropdown to review previously uploaded data.
- **Clean up**: Automatically deletes extracted files upon error or when you upload a new report.
## Visualization

- **Stacked Bar Plot**: Shows the combined frequencies of different actions for each board texture.
- **Equity Distribution**: Displays stacked bar charts that visualize the distribution of equity for both IP and OOP players across various equity buckets (Weak, Okay, Good, Nut).


## How It Works
- **ZIP Extraction**: The application extracts .csv files from uploaded .zip files, and identifies the report files for both IP and OOP players.
- **Board Classification**: Uses a custom classification system to identify the type of board (e.g., Monotone, Two Broadway, one low card).
- **Dynamic Visualization**: Generates interactive stacked bar plots for action frequencies and equity buckets, which are dynamically updated based on selected filters.
