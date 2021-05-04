import jinja2
import numpy as np
import hail as hl
import plotly
import plotly.express as px
import json
from aiohttp import web
import aiohttp_jinja2

app = web.Application()
routes = web.RouteTableDef()

if not hl.hadoop_exists('bn.mt'):
    # Generate data for demonstratation purposes, this should already exist
    mt = hl.balding_nichols_model(
        5,
        100,
        10000,
        pop_dist=[0.1, 0.2, 0.3, 0.2, 0.2],
        fst=[.02, .06, .04, .12, .08],
        af_dist=hl.rand_beta(a=0.01, b=2.0, lower=0.05, upper=1.0),
        mixture=True)
    mt = hl.variant_qc(mt)
    mt.write('bn.mt', overwrite=True)

mt = hl.read_matrix_table('bn.mt')


if not hl.hadoop_exists('scores.t'):
    # Generate data for demonstratation purposes, this should already exist
    scores = hl.hwe_normalized_pca(mt.GT, k=5)[1]
    scores = scores.annotate(**mt.cols()[scores.sample_idx])
    scores.write('scores.t')


pcs = hl.read_table('scores.t')


@routes.get('')
@routes.get('/')
async def get_sha(request):  # pylint: disable=unused-argument
    arr = pcs.collect()
    pca_plot = px.scatter_3d([
        {'id': x['sample_idx'],
         'pop': np.argmax(x['pop']),
         **{f'PC{i}': x['scores'][i] for i in range(5)}}
        for x in arr
    ], x='PC0', y='PC1', z='PC2', color='pop')

    pca_table = pcs._show(n=100, width=None, truncate=None, types=True)._repr_html_()


    afs = mt.variant_qc['AF'][1].collect()
    aaf_plot = px.histogram(afs)

    rows = mt.rows()
    rows = rows.select(**rows.variant_qc)
    af_table = rows._show(n=100, width=None, truncate=None, types=True)._repr_html_()

    graphs = {
        'Top 3 Principal Components': json.dumps(pca_plot, cls=plotly.utils.PlotlyJSONEncoder),
        'Alternate Allele Frequency': json.dumps(aaf_plot, cls=plotly.utils.PlotlyJSONEncoder),
    }
    context = {
        'dataset_name': 'HGDP + 1KG',
        'summary': 'Proin sit amet justo nibh. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla facilisi. In scelerisque rutrum dictum. Donec viverra eros in blandit lobortis. Aenean cursus lorem nec justo aliquam consectetur. Vivamus non risus vitae velit gravida aliquam in at libero. Morbi tristique sem vitae massa laoreet laoreet. Etiam orci erat, consequat ac porta a, imperdiet id nunc. Pellentesque in finibus sapien.',
        'ids': list(graphs.keys()),
        'graphs': list(graphs.values()),
        'tables': [pca_table, af_table],
        'zip': zip
    }
    return aiohttp_jinja2.render_template('index.html', request, context)


aiohttp_jinja2.setup(app, loader=jinja2.PackageLoader('explore'))
app.add_routes(routes)
web.run_app(app, host='0.0.0.0', port=5000)
