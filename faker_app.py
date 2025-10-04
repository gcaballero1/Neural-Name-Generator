from dash import Dash, html, callback, Output, Input
from faker import Faker

fake = Faker()
app = Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("Random Name Generator"),
    html.Button("Generate Names", id="generate-button", n_clicks=0),
    html.Div(id="name-output")
])

@callback(Output("name-output", "children"),
          Input("generate-button", "n_clicks"))
def generate_random_names(n_clicks):
    names = [fake.name() for _ in range(5)]
    return "Random Names: " + ", ".join(names)

if __name__ == "__main__":
    app.run(port=8050, debug=True)
