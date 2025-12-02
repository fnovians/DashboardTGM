import base64
import io
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Inisialisasi aplikasi dengan tema LUX dan supresi error
app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.LUX, "https://fonts.googleapis.com/css2?family=Source+Serif+Pro:wght@400;600;700&display=swap"],
                suppress_callback_exceptions=True)
app.title = "Laboratorium Analisis Kuantitatif"

# --- Layout Aplikasi Utama ---
app.layout = html.Div(style={'fontFamily': '"Source Serif Pro", serif'}, children=[
    dbc.Container(fluid=True, className="py-4 px-5", children=[
        dcc.Store(id='original-data-store'),
        dcc.Store(id='filtered-data-store'),
        dcc.Store(id='error-store'),
        dcc.Store(id='tgm-calculated-data-store'),

        html.Div([
            html.H1("Laboratorium Analisis Kuantitatif", className="display-4"),
            html.P("Sebuah lingkungan interaktif untuk eksplorasi data dan pemodelan matematis.", className="lead text-muted")
        ], className="p-5 mb-4 bg-light border rounded-3"),

        dbc.Row([
            dbc.Col(width=3, children=[
                dbc.Card(className="p-3", children=[
                    html.H4("Panel Kontrol", className="card-title mb-3"),
                    html.Hr(className="my-2"),
                    dbc.Label("1. Unggah Dataset (CSV):", className="fw-bold mb-2"),
                    dcc.Upload(id='upload-data', children=html.Div(['Seret & Lepas atau ', html.A('Pilih File')]), className="border border-dashed rounded p-3 text-center mb-4"),
                    dbc.Label("2. Opsi Delimiter:", className="fw-bold mb-2"),
                    dbc.RadioItems(id='csv-delimiter', options=[{'label': 'Koma (,)', 'value': ','}, {'label': 'Semicolon (;)', 'value': ';'}], value=',', inline=True, className="mb-3"),
                    
                    html.Div(id='filter-panel', style={'display': 'none'}, children=[
                        html.Hr(),
                        dbc.Label("3. Filter Data:", className="fw-bold mb-2"),
                        dbc.Label("Pilih Kolom Tahun:"),
                        dcc.Dropdown(id='year-column-selector', placeholder="Pilih kolom...", className="mb-3"),
                        dbc.Label("Pilih Tahun:"),
                        dcc.Dropdown(id='year-filter-dropdown', placeholder="Pilih tahun...", disabled=True),
                    ])
                ])
            ]),
            dbc.Col(width=9, children=[
                html.Div(id='upload-error-output'),
                html.Div(id="main-content-area", children=dbc.Card(
                    dbc.CardBody(html.H4("Menunggu dataset untuk dianalisis...", className="text-center text-muted py-5")),
                    className="h-100"
                ))
            ])
        ])
    ])
])

# --- Fungsi Parsing ---
def parse_contents(contents, filename, delimiter):
    if contents is None: return None, "Tidak ada file yang diunggah."
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep=delimiter)
        return df.to_json(date_format='iso', orient='split'), None
    except Exception as e:
        return None, f"Gagal mem-parsing file. Error: {e}. Coba ganti delimiter."

# --- CALLBACKS ---

# 1. Handle upload
@app.callback(Output('original-data-store', 'data'), Output('error-store', 'data'), Input('upload-data', 'contents'), State('upload-data', 'filename'), State('csv-delimiter', 'value'))
def handle_upload(contents, filename, delimiter):
    if contents:
        data, error_message = parse_contents(contents, filename, delimiter)
        return data, error_message
    return None, None

# 2. Tampilkan error
@app.callback(Output('upload-error-output', 'children'), Input('error-store', 'data'))
def display_error(error_message):
    if error_message:
        return dbc.Alert(error_message, color="danger", dismissable=True, className="mb-3")
    return None

# 3. Tampilkan panel filter
@app.callback(Output('filter-panel', 'style'), Output('year-column-selector', 'options'), Input('original-data-store', 'data'))
def show_filter_panel(jsonified_data):
    if jsonified_data:
        df = pd.read_json(io.StringIO(jsonified_data), orient='split')
        return {'display': 'block'}, [{'label': col, 'value': col} for col in df.columns]
    return {'display': 'none'}, []

# 4. Isi dropdown filter tahun
@app.callback(Output('year-filter-dropdown', 'options'), Output('year-filter-dropdown', 'disabled'), Output('year-filter-dropdown', 'value'), Input('year-column-selector', 'value'), State('original-data-store', 'data'))
def populate_year_filter(selected_year_col, jsonified_data):
    if selected_year_col and jsonified_data:
        df = pd.read_json(io.StringIO(jsonified_data), orient='split')
        years = sorted(df[selected_year_col].unique())
        options = [{'label': 'Semua Tahun', 'value': 'all'}] + [{'label': str(y), 'value': y} for y in years]
        return options, False, 'all'
    return [], True, None

# 5. Lakukan filtering
@app.callback(Output('filtered-data-store', 'data'), Input('year-filter-dropdown', 'value'), State('year-column-selector', 'value'), State('original-data-store', 'data'))
def filter_data_by_year(selected_year, year_col, jsonified_data):
    if not jsonified_data: return None
    df = pd.read_json(io.StringIO(jsonified_data), orient='split')
    if year_col and selected_year and selected_year != 'all':
        try:
            df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
            selected_year = int(selected_year)
        except (ValueError, TypeError):
            pass
        filtered_df = df[df[year_col] == selected_year]
        return filtered_df.to_json(date_format='iso', orient='split')
    return df.to_json(date_format='iso', orient='split')

# 6. Render Konten Utama (Tabs)
@app.callback(Output('main-content-area', 'children'), Input('filtered-data-store', 'data'))
def render_main_content(jsonified_filtered_data):
    if not jsonified_filtered_data:
        return dbc.Card(dbc.CardBody(html.H4("Menunggu dataset...", className="text-center text-muted py-5")))

    df = pd.read_json(io.StringIO(jsonified_filtered_data), orient='split')
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    tab_data_view = dbc.Card(dbc.CardBody([dash_table.DataTable(data=df.to_dict('records'), columns=[{'name': i, 'id': i} for i in df.columns], page_size=10, style_table={'overflowX': 'auto'}, style_header={'fontWeight': 'bold'}, style_cell={'fontFamily': 'sans-serif', 'textAlign': 'left'}, style_as_list_view=True)], className="p-4"))
    stats_df = df.describe().reset_index().round(2)
    tab_stats = dbc.Card(dbc.CardBody([dash_table.DataTable(data=stats_df.to_dict('records'), columns=[{'name': i, 'id': i} for i in stats_df.columns], style_table={'overflowX': 'auto'}, style_header={'fontWeight': 'bold'}, style_cell={'fontFamily': 'sans-serif', 'textAlign': 'left'}, style_as_list_view=True)], className="p-4"))
    tab_visuals = dbc.Card(dbc.CardBody([dbc.Row([dbc.Col(md=6, children=[dbc.Card([dbc.CardHeader("Distribusi Data (Histogram)"), dbc.CardBody([dbc.Label("Pilih Variabel:", className="fw-bold"), dcc.Dropdown(id='hist-dropdown', options=numeric_cols, value=numeric_cols[0] if numeric_cols else None), dcc.Loading(dcc.Graph(id='histogram-graph', className="mt-3"))])])]), dbc.Col(md=6, children=[dbc.Card([dbc.CardHeader("Hubungan Antar Variabel (Scatter Plot)"), dbc.CardBody([dbc.Label("Sumbu X:", className="fw-bold"), dcc.Dropdown(id='scatter-x-dropdown', options=numeric_cols, value=numeric_cols[0] if numeric_cols else None, className="mb-2"), dbc.Label("Sumbu Y:", className="fw-bold"), dcc.Dropdown(id='scatter-y-dropdown', options=numeric_cols, value=numeric_cols[1] if len(numeric_cols) > 1 else (numeric_cols[0] if numeric_cols else None)), dcc.Loading(dcc.Graph(id='scatter-graph', className="mt-3"))])])])], className="mb-4"), dbc.Row(dbc.Col(dbc.Card([dbc.CardHeader("Matriks Korelasi"), dbc.CardBody(dcc.Loading(dcc.Graph(id='correlation-heatmap-graph')))])))], className="p-4"))
    tab_ml = dbc.Card(dbc.CardBody([dbc.Row([dbc.Col(md=6, children=[html.H5("1. Konfigurasi Model"), dbc.Label("Variabel Target (Y):", className="fw-bold"), dcc.Dropdown(id='ml-target-var', options=numeric_cols, value=numeric_cols[-1] if numeric_cols else None, className="mb-3"), dbc.Label("Variabel Fitur (X):", className="fw-bold"), dcc.Dropdown(id='ml-feature-vars', options=numeric_cols, multi=True, className="mb-3"), dbc.Label("Pilih Model:", className="fw-bold"), dcc.Dropdown(id='ml-model-type', options=['Linear Regression', 'Random Forest'], value='Linear Regression')]), dbc.Col(md=6, className="d-flex flex-column justify-content-center", children=[html.H5("2. Eksekusi"), dbc.Button("Latih & Evaluasi Model", id="ml-train-btn", color="primary", size="lg", className="w-100 mt-4 fw-bold")])]), html.Hr(className="my-4"), html.H5("3. Hasil Evaluasi Model"), dcc.Loading(html.Div(id='ml-results-output'))], className="p-4"))
    tab_tgm = dbc.Card(dbc.CardBody([html.H5("Kalkulator Tingkat Minat Membaca (TGM)"), html.P("Petakan kolom dari dataset Anda ke variabel TGM yang sesuai.", className="text-muted"), html.Hr(), dbc.Row([dbc.Col(md=4, children=[dbc.Label("Frekuensi Membaca (RF):", className="fw-bold"), dcc.Dropdown(id='tgm-rf-col', options=numeric_cols, placeholder="Pilih kolom...")]), dbc.Col(md=4, children=[dbc.Label("Durasi Membaca Harian (DRD):", className="fw-bold"), dcc.Dropdown(id='tgm-drd-col', options=numeric_cols, placeholder="Pilih kolom...")]), dbc.Col(md=4, children=[dbc.Label("Jumlah Membaca (NR):", className="fw-bold"), dcc.Dropdown(id='tgm-nr-col', options=numeric_cols, placeholder="Pilih kolom...")])], className="mb-3"), dbc.Row([dbc.Col(md=4, children=[dbc.Label("Frekuensi Akses Internet (IAF):", className="fw-bold"), dcc.Dropdown(id='tgm-iaf-col', options=numeric_cols, placeholder="Pilih kolom...")]), dbc.Col(md=4, children=[dbc.Label("Durasi Internet Harian (DID):", className="fw-bold"), dcc.Dropdown(id='tgm-did-col', options=numeric_cols, placeholder="Pilih kolom...")]), dbc.Col(md=4, className="d-flex align-items-end", children=[dbc.Button("Hitung TGM", id="tgm-calculate-btn", color="success", className="w-100 fw-bold")])]), html.Hr(className="my-4"), dcc.Loading(html.Div(id='tgm-results-output'))], className="p-4"))
    return dbc.Tabs([dbc.Tab(tab_data_view, label="🔢 Dataset"), dbc.Tab(tab_stats, label="Σ Statistik Deskriptif"), dbc.Tab(tab_visuals, label="📊 Visualisasi"), dbc.Tab(tab_ml, label="🧠 Pemodelan f(x)"), dbc.Tab(tab_tgm, label="📚 TGM")])

# --- Callbacks ---

@app.callback(Output('correlation-heatmap-graph', 'figure'), Input('filtered-data-store', 'data'))
def update_correlation_heatmap(jsonified_data):
    if not jsonified_data: return go.Figure()
    df = pd.read_json(io.StringIO(jsonified_data), orient='split')
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if len(numeric_cols) < 2: return go.Figure().update_layout(title="Butuh minimal 2 kolom numerik untuk korelasi")
    corr_matrix = df[numeric_cols].corr()
    return px.imshow(corr_matrix, text_auto=True, title="Heatmap Korelasi")

@app.callback(Output('histogram-graph', 'figure'), Input('hist-dropdown', 'value'), State('filtered-data-store', 'data'))
def update_histogram(selected_col, jsonified_data):
    if not all([selected_col, jsonified_data]): return go.Figure()
    df = pd.read_json(io.StringIO(jsonified_data), orient='split')
    return px.histogram(df, x=selected_col, title=f"Distribusi Variabel {selected_col}")

@app.callback(Output('scatter-graph', 'figure'), [Input('scatter-x-dropdown', 'value'), Input('scatter-y-dropdown', 'value')], State('filtered-data-store', 'data'))
def update_scatter(x_col, y_col, jsonified_data):
    if not all([x_col, y_col, jsonified_data]): return go.Figure()
    df = pd.read_json(io.StringIO(jsonified_data), orient='split')
    return px.scatter(df, x=x_col, y=y_col, title=f"Hubungan: {x_col} vs {y_col}")

@app.callback(Output('ml-feature-vars', 'options'), Output('ml-feature-vars', 'value'), Input('ml-target-var', 'value'), State('filtered-data-store', 'data'))
def update_feature_options(target_var, jsonified_data):
    if not all([target_var, jsonified_data]): return [], []
    df = pd.read_json(io.StringIO(jsonified_data), orient='split')
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    available_features = [col for col in numeric_cols if col != target_var]
    return available_features, []

@app.callback(Output('ml-results-output', 'children'), Input('ml-train-btn', 'n_clicks'), [State('ml-target-var', 'value'), State('ml-feature-vars', 'value'), State('ml-model-type', 'value'), State('filtered-data-store', 'data')])
def train_ml_model(n_clicks, target_var, feature_vars, model_type, jsonified_data):
    if n_clicks is None: return dbc.Alert("Konfigurasi model Anda dan klik tombol 'Latih & Evaluasi'.", color="info", className="mt-4")
    if not all([target_var, feature_vars, model_type, jsonified_data]): return dbc.Alert("Pastikan semua parameter (Target, Fitur, Model) sudah dipilih.", color="warning", className="mt-4")
    
    df = pd.read_json(io.StringIO(jsonified_data), orient='split')
    
    # --- PERBAIKAN DI SINI ---
    # 1. Pembersihan Data (Data Cleaning)
    modeling_cols = feature_vars + [target_var]
    original_rows = len(df)
    df_cleaned = df[modeling_cols].dropna()
    cleaned_rows = len(df_cleaned)
    dropped_rows = original_rows - cleaned_rows
    
    # 2. Safety Check untuk ukuran data setelah dibersihkan
    if cleaned_rows < 10:
        return dbc.Alert(f"Dataset terlalu kecil untuk pemodelan setelah {dropped_rows} baris dengan data kosong dihapus. Tersisa {cleaned_rows} baris. Coba pilih 'Semua Tahun' atau tahun dengan data lebih banyak.", color="warning", className="mt-4")

    X = df_cleaned[feature_vars]
    y = df_cleaned[target_var]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression() if model_type == 'Linear Regression' else RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test, y=predictions, mode='markers', name='Prediksi vs. Aktual'))
    fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], mode='lines', name='Garis Ideal y=x', line=dict(dash='dash', color='black')))
    fig.update_layout(title='Evaluasi Model: Prediksi vs. Nilai Aktual', xaxis_title='Nilai Aktual', yaxis_title='Nilai Prediksi')
    
    # 3. Notifikasi transparan
    cleaning_notification = dbc.Alert(f"Info: {dropped_rows} baris dengan nilai kosong telah diabaikan dari {original_rows} total baris.", color="info")

    return html.Div([
        cleaning_notification,
        dbc.Alert(f"Model '{model_type}' berhasil dievaluasi! Mean Squared Error (MSE): {mse:.2f}", color="success", className="mt-2"),
        dcc.Graph(figure=fig)
    ])

@app.callback(Output('tgm-results-output', 'children'), Output('tgm-calculated-data-store', 'data'), Input('tgm-calculate-btn', 'n_clicks'), [State('tgm-rf-col', 'value'), State('tgm-drd-col', 'value'), State('tgm-nr-col', 'value'), State('tgm-iaf-col', 'value'), State('tgm-did-col', 'value'), State('filtered-data-store', 'data')])
def calculate_tgm(n_clicks, rf_col, drd_col, nr_col, iaf_col, did_col, jsonified_data):
    if n_clicks is None: return dbc.Alert("Pilih semua kolom yang sesuai dan klik 'Hitung TGM'.", color="info"), None
    if not all([rf_col, drd_col, nr_col, iaf_col, did_col]): return dbc.Alert("Error: Pastikan kelima kolom telah dipilih.", color="danger"), None
    df = pd.read_json(io.StringIO(jsonified_data), orient='split')
    df['TGM_Score'] = (0.3 * df[rf_col] + 0.3 * df[drd_col] + 0.3 * df[nr_col]) + (0.05 * df[iaf_col] + 0.05 * df[did_col])
    bins = [0, 20, 40, 60, 80, 101]; labels = ['Sangat Rendah', 'Rendah', 'Moderat', 'Tinggi', 'Sangat Tinggi']
    df['TGM_Kategori'] = pd.cut(df['TGM_Score'], bins=bins, labels=labels, right=False, include_lowest=True)
    category_counts = df['TGM_Kategori'].value_counts().reindex(labels)
    fig_bar = px.bar(x=category_counts.index, y=category_counts.values, title='Distribusi Kategori TGM', labels={'x': 'Kategori TGM', 'y': 'Jumlah Responden'})
    scatter_options = [{'label': col, 'value': col} for col in [rf_col, drd_col, nr_col, iaf_col, did_col]]
    output_layout = html.Div([dbc.Row([dbc.Col(md=6, children=[dbc.Card([dbc.CardHeader("1. Distribusi Kategori TGM"), dbc.CardBody(dcc.Graph(figure=fig_bar))])]), dbc.Col(md=6, children=[dbc.Card([dbc.CardHeader("2. Analisis Hubungan TGM"), dbc.CardBody([dbc.Label("Pilih Variabel untuk dianalisis:", className="fw-bold"), dcc.Dropdown(id='tgm-scatter-var-dropdown', options=scatter_options, value=rf_col), dcc.Graph(id='tgm-scatter-graph', className="mt-3")])])])], className="mb-4"), dbc.Row(dbc.Col(dbc.Card([dbc.CardHeader("3. Tabel Hasil Perhitungan"), dbc.CardBody(dash_table.DataTable(data=df.to_dict('records'), columns=[{'name': i, 'id': i} for i in df.columns], page_size=5, style_table={'overflowX': 'auto'}, style_header={'fontWeight': 'bold'}, style_cell={'fontFamily': 'sans-serif', 'textAlign': 'left'}, style_as_list_view=True))])))])
    return output_layout, df.to_json(date_format='iso', orient='split')

@app.callback(Output('tgm-scatter-graph', 'figure'), Input('tgm-scatter-var-dropdown', 'value'), State('tgm-calculated-data-store', 'data'))
def update_tgm_scatter(selected_var, jsonified_tgm_data):
    if not all([selected_var, jsonified_tgm_data]): return go.Figure()
    df = pd.read_json(io.StringIO(jsonified_tgm_data), orient='split')
    fig = px.scatter(df, x=selected_var, y='TGM_Score', title=f'Hubungan: {selected_var} vs. Skor TGM', labels={selected_var: f'Nilai {selected_var}', 'TGM_Score': 'Skor TGM'})
    return fig

# --- Menjalankan server ---
if __name__ == '__main__':
    app.run(debug=True)
