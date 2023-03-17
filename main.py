import json
import io
import logging
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main(df, tree):

    # on first run, default like so
    x_center = np.mean([df['pca_0'].min(), df['pca_0'].max()])
    y_center = np.mean([df['pca_1'].min(), df['pca_1'].max()])
    if 'x' not in st.session_state:
        st.session_state['x'] = x_center
    if 'y' not in st.session_state:
        st.session_state['y'] = y_center

    st.title('Galaxy Zoo Explorer')
    st.subheader('by Mike Walmsley ([@mike\_walmsley\_](https://twitter.com/mike_walmsley_))')

    st.markdown('---')

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('## Move Around')
        st.markdown('Click anywhere on the latent space to view galaxies')
        with st.empty():
            new_coords = show_explorer(df) # will read current state
            if new_coords:
            # if we had a click, update the state
                st.session_state['x'] = new_coords[0]
                st.session_state['y'] = new_coords[1]
                # rerun
                show_explorer(df)
        # st.markdown(st.session_state['x'])
        # st.markdown(st.session_state['y'])
        # if no click (i.e. on first run), will not update the state

    with col2:
        # read the state 
        galaxies = get_closest_galaxies(
            tree,
            st.session_state['x'],
            st.session_state['y'],
            max_neighbours=12
        )
        show_gallery_of_galaxies(galaxies)
    
    st.markdown('---')
    with st.expander('Tell me more'):
        tell_me_more()
    

def tell_me_more():
    st.markdown("""
    Here, I may explain some things - but not yet.
    """)



def show_explorer(df):
    return show_latent_space_interface(df)



def show_latent_space_interface(df):

    fig, ax = plot_latent_space(df, st.session_state['x'], st.session_state['y'])
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='jpg', dpi=100)
    buf.seek(0)
    im = Image.open(buf)
    # im.show()
    # buf.close()

    # https://github.com/blackary/streamlit-image-coordinates
    x_y_dict = streamlit_image_coordinates(im) # None or {x: x, y: y}

    if x_y_dict is None:  # no click yet, return None
        return None
    else:
        # was a click, return as coordinates
        x_pix, y_pix = x_y_dict['x'], x_y_dict['y']
        x, y = ax.transData.inverted().transform([x_pix, y_pix])
        y = -y+.5  # matplotlib coordinates
        return x, y


def plot_latent_space(df, x=None, y=None, figsize=(4, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(df['pca_0'], df['pca_1'], s=.3, alpha=.07)
    if x is not None:
        ax.scatter(x, y, marker='+', c='r')
    plt.axis('off')
    return fig, ax

def get_closest_galaxies(tree, x, y, max_neighbours=1):
    # st.markdown(x)
    # st.markdown(y)
    _, indices = tree.kneighbors(np.array([x, y]).reshape(1, 2))
    indices = indices[0]  # will be wrapped in extra dim
    return df.iloc[indices[:max_neighbours]]


def show_gallery_of_galaxies(df):

    image_urls = df.apply(get_galaxy_url, axis=1)
    # for url in image_urls:
    #     st.image(url)

    opening_html = '<div style=display:flex;flex-wrap:wrap>'
    closing_html = '</div>'
    child_html = ['<img src="{}" style=margin:3px;width:200px;></img>'.format(url) for url in image_urls]

    gallery_html = opening_html
    for child in child_html:
        gallery_html += child
    gallery_html += closing_html

    st.markdown(gallery_html, unsafe_allow_html=True)

def get_galaxy_url(row):
    field_of_view_pixscale = row['est_dr5_pixscale']
    max_native_pixels = 512

    historical_size = 424
    arcsecs = historical_size * field_of_view_pixscale

    native_pixscale = 0.262
    pixel_extent = np.ceil(arcsecs / native_pixscale).astype(int)

    ra = row['ra']
    dec = row['dec']
    data_release = '8'
    max_size = max_native_pixels
    img_format = 'jpg'

    # Two cases. Either galaxy is extended beyond maxsize, in which case download at maxsize and dif scale,
    # or (more often) it's small, and everything's okay
    params = {
        'ra': ra,
        'dec': dec,
        'layer': 'dr{}&bands=grz'.format(data_release)
    }

    if pixel_extent < max_size:
        params['size'] = pixel_extent
        query_params = 'ra={}&dec={}&size={}&layer={}'.format(
            params['ra'], params['dec'], params['size'], params['layer'])
    else:
        # forced to rescale to keep galaxy to reasonable number of pixels
        pixel_scale = arcsecs / max_size
        params['size'] = max_size
        params['pixscale'] = pixel_scale
        query_params = 'ra={}&dec={}&size={}&layer={}&pixscale={}'.format(
            params['ra'], params['dec'], params['size'], params['layer'], params['pixscale'])

    if img_format == 'jpg':
        img_format = 'jpeg'

    url = "http://legacysurvey.org/viewer/{}-cutout?{}".format(img_format, query_params)

    return url




st.set_page_config(
    layout="wide",
    page_title='GZ DESI',
    page_icon='gz_icon.jpeg'
)

@st.cache_resource
def load_data():
    df = pd.read_parquet('/Users/user/repos/galaxy-explorer-streamlit/50k_pca_2n.parquet')

    X = df[['pca_0', 'pca_1']]
    # will always return 100 neighbours, cut list when used
    nbrs = NearestNeighbors(n_neighbors=100, algorithm='ball_tree').fit(X)

    return df, nbrs
    # df_locs = ['decals_{}.csv'.format(n) for n in range(4)]
    # dfs = [pd.read_csv(df_loc) for df_loc in df_locs]
    # return pd.concat(dfs)

if __name__ == '__main__':

    logging.basicConfig(level=logging.CRITICAL)

    df, tree = load_data()

    main(df, tree)
