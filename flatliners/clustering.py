from .baseflatliner import BaseFlatliner
import logging
import time
import sys

import ocp.jupyter as oj
import pandas as pd
import numpy as np

from sklearn.preprocessing import PowerTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from umap import UMAP


# setup logging
_LOGGER = logging.getLogger(__name__)


# TODO: remove pickle, time imports
# TODO: ensure time interval is 270s
# TODO: add input sanitization
# TODO: add version type, duration
# TODO: make convert functions and logging tag static
# TODO: make umap learn w/ failure semisupervision
class Clusterer(BaseFlatliner):

    # querying handle for global queries (not specific to a deployment id)
    _all_depls_querier = oj.global_queries

    def __init__(self, num_nearest=10):
        super().__init__()
        # how many nearest clusters to pubish
        self.num_nearest = num_nearest

        # data transformers and clusertering model
        self._power_transf = PowerTransformer()
        self._umap_transf = UMAP(n_components=3, metric='hamming', n_neighbors=100, min_dist=0.1, random_state=42)
        self._model = KMeans(n_clusters=8, max_iter=1000, n_jobs=-1, random_state=42)

    def on_next(self, x):
        # TODO: perform only for the concerned version
        # setup data in a "data science friendly" / "sklearn friendly" data structure
        conditions_df = self._opconds_metrics_to_df(x['cluster_operator_conditions'])
        install_df = self._installer_metrics_to_df(x['cluster_installer'])
        metrics_df = conditions_df.merge(install_df, how='left', left_index=True, right_index=True)

        # nans because some install types are neither upi nor ipi (unknown)
        metrics_df['install_type_IPI'] = metrics_df['install_type_IPI'].fillna(0)
        metrics_df['install_type_UPI'] = metrics_df['install_type_UPI'].fillna(0)

        # train model
        self._model.fit(
            self._umap_transf.fit_transform(
                X=self._power_transf.fit_transform(metrics_df),
                y=None
            )
        )

        # publish "nearest" deployments for each deployment within cluster id
        ts = time.time()
        for cid in np.unique(self._model.labels_):
            # transformed data for deployments assigned to cluster `cid`
            cid_metrics = self._umap_transf.transform(
                                self._power_transf.transform(
                                    metrics_df.iloc[self._model.labels_ == cid]
                                )
                            )

            # number of nearest neighbors cant be more than number of deployments in cluster
            cid_nn = min(1+self.num_nearest, len(cid_metrics))
            _LOGGER.info("Clusterer: Finding {} nearest deployments for deployments (n={}) of cluster id {}".format(cid_nn, cid_metrics.shape[0], cid))

            # fit nearest neighbor querier and get nearest
            nearneigh = NearestNeighbors(n_neighbors=cid_nn, n_jobs=-1).fit(cid_metrics)
            kn_idx = nearneigh.kneighbors(cid_metrics, return_distance=False)

            # get deployment id from index into cid_metrics
            cid_all_ids = metrics_df.iloc[self._model.labels_ == cid].index
            depl_id_from_idx = np.vectorize(lambda i: cid_all_ids[i])
            kn_depl_ids = depl_id_from_idx(kn_idx)

            _LOGGER.info("Clusterer: Publishing nearest deployments for cluster id {}".format(cid))
            self._publish_nearest_depls(kn_depl_ids, ts)

    @staticmethod
    def _opconds_metrics_to_df(metrics_raw):
        # convert to dataframe and fix dtype inference
        df = Clusterer._all_depls_querier.metric_to_dataframe(metrics_raw)
        df['value'] = df['value'].astype(np.float64)

        # clayton suggested filling data with empty string
        # however doing that will make it difficult to do some pandas column name operations
        # so fill it with the string "empty"
        df['reason'] = df['reason'].fillna("empty")

        # convert to one-hot encoding
        opcond_keep_cols = ['_id', 'name', 'condition', 'value', 'reason']
        onehot_df = df[opcond_keep_cols].groupby(['_id', 'name', 'condition', 'reason']).first()

        # make each condition for each operator a column in the data
        onehot_df = onehot_df.unstack(level=[lvl_idx for lvl_idx in range(1, onehot_df.index.nlevels)])

        # ensure no deployment_ids were lost in translation (literally, "translation" of the matrix)
        assert onehot_df.index.nunique()==df['_id'].nunique()

        # fill nans
        onehot_df = onehot_df.fillna(0)

        # convert multi-dimensional column indexing into single dimnsion
        new_colnames = ['_'.join((op, cond, reason))
                        for op,cond,reason in zip(onehot_df.columns.get_level_values(1),
                                                    onehot_df.columns.get_level_values(2),
                                                    onehot_df.columns.get_level_values(3))]
        onehot_df.columns = new_colnames

        return onehot_df

    @staticmethod
    def _installer_metrics_to_df(metrics_raw):
        # this seems to be how labels are assigned, according to the query on grafana
        # label_replace(label_replace(cluster_installer{_id="$_id"}, "type", "UPI", "type", ""), "type", "IPI", "type", "openshift-install")
        df = Clusterer._all_depls_querier.metric_to_dataframe(metrics_raw)
        df['type'] = df['type'].replace(to_replace=[np.nan, 'openshift-install'], value=['UPI', 'IPI'])

        # ensure that ids are unique. then make id the index into df
        df = df.drop_duplicates(subset=['_id', 'version', 'type'])
        assert df['_id'].nunique()==len(df)
        df = df.set_index(keys='_id')

        # keep only relevant columns
        return pd.get_dummies(df[['type']], prefix='install_type')

    def _publish_nearest_depls(self, kn_depl_ids, timestamp):
        """Publishes Prometheus metric (nearest_deployments) for each deployment
        (n) in kn_depl_ids

        Arguments:
            kn_depl_ids {np.ndarray} -- ndarray of shape (n x k) where n is number
                of deployments and k is the number of nearest neighbors plus one
            timestamp {float} -- "value" of the prometheus metric
        """
        # the first element in kn_depl_ids is always the depl itself
        curr_num_nearest = kn_depl_ids.shape[1]-1
        for i in range(len(kn_depl_ids)):
            # create prom metric for each depl
            # FIXME: need a way to pass versions
            metric = self.Nearest_Deployments_Metric(
                num = curr_num_nearest,
                _id = kn_depl_ids[i, 0],
                version = 'test',
                timestamp = timestamp,
            )
            # add k nearest neighbors
            for ni, nid in enumerate(kn_depl_ids[i, 1:]):
                metric.nearest_deployments[ni] = nid
            self.publish(metric)

    class Nearest_Deployments_Metric:
        def __init__(self, num: int, timestamp: float, _id: str, version: str, nearest_deployments: dict = None):
            self.num = num
            self._id = _id
            self.version = version
            self.timestamp = timestamp
            if nearest_deployments is None:
                self.nearest_deployments = dict((i, "") for i in range(num))


class ClusteringMetricsGatherer(BaseFlatliner):
    def __init__(self, metric_list=('cluster_operator_conditions', 'cluster_installer'), buffer_seconds=270):
        super().__init__()
        # time interval at which CVO sends metrics
        self.buffer_seconds = buffer_seconds

        # timestamp when first (oldest) metric in "current" stream was received
        self._stream_oldest_ts = sys.maxsize

        # prometheus metrics used as features for clustering
        self.metric_list = metric_list

        # datastore dict. key=(metric name), value=(this metric's data for all deployments)
        self._clustering_metrics = {metric: [] for metric in self.metric_list}

    def on_next(self, x):
        curr_metric_name = self.metric_name(x)

        # early exit if this metric is not a feature
        if curr_metric_name not in self.metric_list:
            return

        # timestamp at which metric was recorded
        curr_ts = self.metric_values(x)[0][0]

        # NOTE: if the stream had been coming in the order of timestamps then
        # this check wasnt needed, and simply setting the timestamp of the first
        # metric as the start time would have worked
        if curr_ts < self._stream_oldest_ts:
            self._stream_oldest_ts = curr_ts
            _LOGGER.info("ClusteringMetricsGatherer: Updated stream start timestamp to {}".format(curr_ts))

        # this will be true when the current metric belongs to the "next" stream
        # and not the "current" stream
        # NOTE: this implies that old metrics are processed when the first of the new metrics is received
        # and not when the last of the old metrics is received
        elif (curr_ts - self._stream_oldest_ts) > self.buffer_seconds:
            _LOGGER.info("ClusteringMetricsGatherer: Publishing current metric stream to observers")
            self.publish(self._clustering_metrics)
            # reset for next stream
            self._clustering_metrics = {metric: [] for metric in self.metric_list}
            self._stream_oldest_ts = curr_ts

        # add to datastore
        self._clustering_metrics[curr_metric_name].append(x)
