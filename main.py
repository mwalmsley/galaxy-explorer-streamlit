import io
import logging
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def update_center_coords(df):
    # on first run, default like so
    x_center = np.mean([df['feat_0'].min(), df['feat_0'].max()])
    y_center = np.mean([df['feat_1'].min(), df['feat_1'].max()])
    if 'x' not in st.session_state:
        st.session_state['x'] = x_center
    if 'y' not in st.session_state:
        st.session_state['y'] = y_center


def main():

    # TODO use tabs to allow more galaxies
    # TODO mke galaxies clickable for id_str with on_click callback

    st.title('Galaxy Zoo Explorer')
    st.subheader('by Mike Walmsley ([@mike\_walmsley\_](https://twitter.com/mike_walmsley_))')

    st.markdown('---')


    col1, col2 = st.columns([1, 2])

    with col1:
        # add these first, as we need to have selectbox already to choose which to load
        st.markdown('## Move Around')
        st.markdown('Click anywhere on the latent space to view galaxies.')
        reduction_name = st.selectbox(
            'Select Reduction', ['Featured v2', 'Featured v1', 'Featured and Smooth']
        )

    df = load_umap(reduction_name)
    update_center_coords(df)
    
    tree = fit_tree(df)

    with col1:
        with st.empty():
            new_coords = show_latent_space_interface(df, reduction_name) # will read current state
            if new_coords:
            # if we had a click, update the state
                st.session_state['x'] = new_coords[0]
                st.session_state['y'] = new_coords[1]
                # rerun
                show_latent_space_interface(df, reduction_name)

        st.markdown('Location: ({:.3f}, {:.3f})'.format(st.session_state['x'], st.session_state['y']))
        # if no click (i.e. on first run), will not update the state

    with col2:
        # read the state 
        galaxies = get_closest_galaxies(
            df,
            tree,
            st.session_state['x'],
            st.session_state['y'],
            max_neighbours=1000
        )
        galaxies['url'] = galaxies.apply(get_galaxy_url, axis=1)
        show_gallery_of_galaxies(galaxies[:12])

        csv = convert_df(galaxies)

        st.download_button(
            label="Download CSV of the 1000 galaxies closest to your search",
            data=csv,
            file_name='closest_1000_galaxies.csv',
            mime='text/csv',
        )

    
    st.markdown('---')
    # with st.expander('Tell me more'):
    tell_me_more()
    

def tell_me_more():
    st.markdown("""
    This explorer visualises the internal representation of [Zoobot](github.com/mwalmsley/zoobot),
    Galaxy Zoo's deep learning model. 
    This version of Zoobot is trained to predict the responses of volunteers from all major Galaxy Zoo projects.

    Zoobot's overparametrized representation is first compressed from 1280 to 30 dimensions with iterative PCA.
    This preserves about 98\% of the variance. 
    The 30-dimensional embedding is then visualised with UMAP
    (set to min_dist=0.01 and n_neighbours=200, to allow clumps and focus on global structure).

    The galaxies shown are those closest to your selected point in latent space.
    The closest galaxies are shown first.
    To save your results and see more than 12 galaxies, press the "Download CSV" button under the images.

    The galaxies are drawn from the DESI Legacy Surveys.
    Only galaxies with redshift below z=0.1 are shown.
    Redshifts are spectroscopic where available from SDSS and photometric otherwise.
    Representations for all DESI-LS galaxies will be available with the upcoming GZ DESI data release.

    - "Featured v2" includes 500k galaxies predicted to have more than half of volunteers respond "Featured". 
    - "Featured v1" is identical but includes only 100k galaxies.
    - "Featured and Smooth" shows 100k galaxies selected randomly i.e. including (and dominated by) smoother galaxies.

    *Thanks to Dustin Lang for creating the DESI-LS cutout service used here to dynamically show the images.
    And of course, thanks to the Galaxy Zoo volunteers who make all of this possible. I hope this is a helpful tool for you.*
    """)


@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    df['brickid'] = df['id_str'].apply(lambda x: x.split('_')[0])
    df['objid'] = df['id_str'].apply(lambda x: x.split('_')[1])
    df = df.rename(columns={'feat_0': 'umap_x', 'feat_1': 'umap_y'})
    # re-order cols
    return df[['id_str', 'brickid', 'objid', 'ra', 'dec', 'est_dr5_pixscale', 'umap_x', 'umap_y', 'url']].to_csv(index=False).encode('utf-8')



def show_latent_space_interface(df, reduction_name):

    fig, ax = plot_latent_space(df, st.session_state['x'], st.session_state['y'])
    fig.tight_layout()

    # https://stackoverflow.com/questions/8598673/how-to-save-a-pylab-figure-into-in-memory-file-which-can-be-read-into-pil-image/8598881
    im = fig_to_pil(fig)

    # https://github.com/blackary/streamlit-image-coordinates
    x_y_dict = streamlit_image_coordinates(im) # None or {x: x, y: y}

    if x_y_dict is None:  # no click yet, return None
        return None
    else:
        # was a click, return as coordinates
        x_pix, y_pix = x_y_dict['x'], x_y_dict['y']
        x, y = ax.transData.inverted().transform([x_pix, y_pix])
        if reduction_name == 'Featured v2':
            offset = 9.4
        elif reduction_name == 'Featured v1':
            offset = 9.8
        elif reduction_name == 'Featured and Smooth':
            offset = 10.
        else:
            raise ValueError(reduction_name)
        y = -y + offset
        return x, y


def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='jpg', dpi=100)
    buf.seek(0)
    return Image.open(buf)


def plot_latent_space(df, x=None, y=None, figsize=(4, 4), data_pad=0.1):
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(df['feat_0'], df['feat_1'], s=.2, alpha=.02)
    if x is not None:
        ax.scatter(x, y, marker='+', c='r')
    plt.xlim(df['feat_0'].min()-data_pad, df['feat_0'].max() + data_pad)
    plt.ylim(df['feat_1'].min()-data_pad, df['feat_1'].max() + data_pad)
    plt.axis('off')
    return fig, ax

def get_closest_galaxies(df, tree, x, y, max_neighbours=1):
    # st.markdown(x)
    # st.markdown(y)
    _, indices = tree.kneighbors(np.array([x, y]).reshape(1, 2))
    indices = indices[0]  # will be wrapped in extra dim
    return df.iloc[indices[:max_neighbours]]


def show_gallery_of_galaxies(df):

    if 'url' in df.columns.values:
        image_urls = df['url']
    else:
        image_urls = df.apply(get_galaxy_url, axis=1)

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
def load_umap(name):
    # df = pd.read_parquet('100k_umap_2n.parquet')
    if name == 'Featured v1':
        df = pd.read_parquet('100k_feat_umap_2n.parquet')
    elif name == 'Featured v2':
        df = pd.read_parquet('all_feat_umap_2n.parquet')
    elif name == 'Featured and Smooth':
        df = pd.read_parquet('100k_umap_2n.parquet')
    else:
        raise ValueError(name)
    umap_cols = ['umap_{}'.format(n) for n in range(2)]
    feat_cols = [col.replace('umap_', 'feat_') for col in umap_cols]
    df = df.rename(columns=dict(zip(umap_cols, feat_cols)))
    return df


def fit_tree(df):
    X = df[['feat_0', 'feat_1']]
    # will always return 100 neighbours, cut list when used
    nbrs = NearestNeighbors(n_neighbors=500, algorithm='ball_tree').fit(X)
    return nbrs

if __name__ == '__main__':

    logging.basicConfig(level=logging.CRITICAL)

    main()
