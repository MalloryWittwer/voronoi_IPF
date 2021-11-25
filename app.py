import numpy as np
import dash
from dash import html, State, dcc
import dash_bootstrap_components as dbc
import dash_daq as daq
from dash.dependencies import Input, Output
from serve_voronoi import get_voronoi, fig_to_uri, fill_plot_outline, set_colorbar, get_region_values, compute_vectors
import matplotlib.pyplot as plt
from scipy.spatial import voronoi_plot_2d

app = dash.Dash(__name__, title='IPF Plot Experiment', 
                external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css', dbc.themes.CYBORG])

server = app.server

EULERS = np.load('eulers_demo.npy')
VALUES = np.load('values_demo.npy')

def serve_layout():
    layout = html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.Img(src="assets/hexagons-white.png", id="tiles-label"),
                    html.Div([
                        dcc.Slider(id="input-tiles", min=100, max=800, step=100, value=400, tooltip={"placement": "bottom", "always_visible": True}),
                    ], id="tiles-container"),
                    html.Img(src="assets/hexagons-white-many.png", id="tiles-label-many"),
                ], id="grid-container-r1"),
                html.Div([  
                    html.Img(src="assets/3d-cube-white.png", id="radios-label"),
                    dcc.RadioItems(
                        options=[
                            {'label': 'X', 'value': 'x'},
                            {'label': 'Y', 'value': 'y'},
                            {'label': 'Z', 'value': 'z'}
                        ], value='z', id="radio-items"),
                ], id="grid-container-r2"),
                html.Div([
                    html.Div([
                        html.Label("Min", htmlFor="min-input", id="min-label"),
                        dcc.Input(type="number", value=0, id="min-input")
                    ], id="min-container"),
                    html.Div([
                        html.Label("Max", htmlFor="max-input", id="max-label"),
                        dcc.Input(type="number", value=20, id="max-input")
                    ], id="max-container"),
                ], id="grid-min-max"),
                html.Div([
                    html.Div([
                        html.Img(src="assets/icon-lw.png", id="lw-label"),
                        daq.BooleanSwitch(id='checklist', on=True),
                    ], id='line-width-container'),
                    dcc.Dropdown(
                        id='colormap-dropdown',
                        options=[
                            {'label': 'Blues', 'value': 'Blues'},
                            {'label': 'Grays', 'value': 'gray'},
                            {'label': 'Viridis', 'value': 'viridis'},
                            {'label': 'Inferno', 'value': 'inferno'},
                        ], value='Blues',
                    ),
                ], id="grid-colormap"),
            ], id="pannel"),
            html.Div([
                dcc.Loading(id="loading", type="default", children=html.Img(src="assets/default_img.png", id="plt-figure"))
            ], id="canvas")
        ], id="content-container"),
        html.Div([
            html.A("About", id="about", n_clicks=0),
            html.A("View code", id="code", href="https://github.com/MalloryWittwer/voronoi_IPF", target="_blank"),
            dbc.Modal([
                dbc.ModalBody("In a face-centered cubic material, the orientation distribution of a generic property of interest (ex: misorientation) of the crystallites is displayed. The value of the property is averaged locally in the orientation space, within discrete bins defined by the stereographic projection of a Fibonacci sphere and confined in the inverse pole figure triangle."),
                html.Img(src="assets/figure-about.png", id="modal-image"),
                dbc.ModalFooter(
                    dbc.Button("Close", id="close", className="mr-auto", n_clicks=0), 
                    id="modal-footer"),
            ], id="modal", is_open=False, backdrop=True, centered=True, fade=True),
        ], id="footer-container"),
    ], id="content-wrapper")
    return layout

app.layout = serve_layout

@app.callback(
    Output("plt-figure", "src"),
    Input("checklist", "on"),
    Input("input-tiles", "value"),
    Input("radio-items", "value"),
    Input('min-input', 'value'),
    Input('max-input', 'value'),
    Input('colormap-dropdown', 'value'),
)
def update_output(chx, nft, dir, vmin, vmax, cm):
        
    fig, ax = plt.subplots(figsize=(7,5), dpi=200)
    fig.patch.set_alpha(0)
    
    vectors = compute_vectors(EULERS, dir) # (x, y) coordinates of eulers
    
    # ax.scatter(x=vectors[:,0], y=vectors[:,1], c=VALUES, cmap=plt.cm.Blues)
    
    lw = 0.5 if chx else 0.0        
    vor = get_voronoi(nft)    
    voronoi_plot_2d(vor, show_points=False, show_vertices=False, line_colors='black', 
                    ax=ax, line_width=lw)
    region_values = get_region_values(vor, vectors, VALUES)
    ax = set_colorbar(region_values, ax, vor, vmin, vmax, cm)

    ax = fill_plot_outline(ax)
    ax.set_ylim(0, 0.40)
    ax.set_xlim(0, 0.45)
    ax.axis('off')
    
    return fig_to_uri(fig)

@app.callback(
    Output("modal", "is_open"),
    [Input("about", "n_clicks"), Input("close", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_hot_reload=False)
