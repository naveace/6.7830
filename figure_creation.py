#%%
import plotly.express as px
import arviz as az
import pymc
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dimension', type=int)
args = parser.parse_args()
dimension = args.dimension # type: ignore
basic_model = pymc.Model()
with basic_model:
    x = pymc.Normal('x', mu=0, sigma=1, shape=dimension)

with basic_model:
    trace = pymc.sample(10_000, tune=1000, cores=1, chains=4)
stacked = az.extract(trace)
xs = stacked['x']
r = np.linalg.norm(xs, axis=0)
# %%
print(f'Dimension {dimension}')
print(f"\t {az.hdi(r, hdi_prob=0.89)}")
#%%
from PIL import Image
from pathlib import Path
prefix_path = '/Users/evanvogelbaum/Downloads/betancourt_fig_{num}.png'
for fig_num in [10, 14, 19]:
    img = Image.open(prefix_path.format(num=fig_num))
    img_resized = img.resize((2000, 800))
    path_to_save = Path(
        '/Users/evanvogelbaum/Downloads/betancourt_fig_{num}_resized.png'.format(num=fig_num)
    )
    img_resized.save(path_to_save)
# %%
img.resize((2000, 800))
# %%
import pandas as pd
df = pd.DataFrame({
    'Dim': [1, 10, 100, 1000, 10_000],
    'HDI Min': [4.58911721e-05,1.98755715,8.86339881, 30.47734057, 98.87338845],
    'HDI Max': [1.60154077e+00, 4.21606504, 11.12701776, 32.7423201, 101.148774]
})
df['HDI Mean'] = (df['HDI Min'] + df['HDI Max']) / 2
df['e-'] = df['HDI Mean'] - df['HDI Min']
df['e+'] = df['HDI Max'] - df['HDI Mean']
import plotly.express as px
fig = px.scatter(
    x=df['Dim'],
    y=df['HDI Mean'],
    error_y_minus=df['e-'],
    error_y=df['e+'],
    log_x=True,
    log_y=True,
    width=2000,
    height=800
)
fig.update_layout(
    xaxis_title='Dimension',
    yaxis_title='89% HDI For Radius',
)
fig.update_traces({'marker': {'size': 0.001}})
fig.update_layout(title="89% HDI For Radius of Multivariate Normal")
fig.update_layout(font=dict(size=20))
fig.show()
fig.write_image('/Users/evanvogelbaum/Downloads/89_hdi_for_radius_of_multivariate_normal.png' )
# %%

# %%
