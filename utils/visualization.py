from sklearn import metrics as sklearn_metrics
import plotly.graph_objects as go
import plotly.figure_factory as ff


class Visualization(object):

    def __init__(self, cfg):
        self.cfg = cfg

    def plot_precision_recall_curve(self, p, r, l, dk2, file_name):

        ps, rs, _ = sklearn_metrics.precision_recall_curve(l, dk2)
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                line=dict(color="#00CED1", width=1),
                name='Sklearn',
                x=rs,
                y=ps))

        fig.add_trace(
            go.Scatter(
                line=dict(color="#900C3F", width=1),
                name='Implemented',
                x=r,
                y=p))

        # fig.write_image(self.cfg.plots_dir + f"{file_name}.png")
        fig.show()

    def plot_conf_matrix(self, conf_matrix_for_best_thr, file_name):

        z = conf_matrix_for_best_thr[::-1]
        x = ['Normal', 'Anomaly']
        y = x[::-1].copy()
        z_text = [[str(y) for y in x] for x in z]

        fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='Viridis')
        fig.update_layout(title_text='<i><b>Confusion matrix</b></i>',
                          xaxis=dict(title='Predicted'),
                          yaxis=dict(title='Actual'))

        fig['data'][0]['showscale'] = True
        # fig.write_image(self.cfg.plots_dir + f"{file_name}.png")
        fig.show()
