"""The top-quark tagging reference data set (Zenodo 2603256), served through the hls4ml
datamodule's pipeline: the same rebase on the kept axis, transforms, folds and test
width, over the cache of `src.data.components.top_tagging`."""

from src.data.components.top_tagging import TopTaggingRecord
from src.data.hls4ml_datamodule import HLS4MLDataModule
from src.symmetry import CLASS_NAMES_OF
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__)


class TopTaggingDataModule(HLS4MLDataModule):
    """Serve the converted top-tagging jets, columns ``px, py, pz, e, pt, eta, phi,
    etarel, phirel``.

    The arguments are those of `HLS4MLDataModule`. ``cache_dir`` holds ``raw/`` and one
    ``nconst_<N>/`` per constituent count, which ``prepare_data`` builds on first use;
    ``revision`` is unused, and ``full_axis`` is refused.
    """

    class_names = CLASS_NAMES_OF["top_tagging"]

    def _ensure_record(self) -> TopTaggingRecord:
        """Open the cache of ``nconstituents`` under ``cache_dir``.

        :raises ValueError: under ``full_axis``: the cache holds no column measured from
            the untruncated jet's axis.
        """
        if self.full_axis:
            raise ValueError(
                "full_axis keeps the axis columns of the untruncated jet, which the "
                "top-tagging cache does not hold: set data.full_axis=false."
            )
        if self.record is None:
            self.record = TopTaggingRecord(self.cache_dir, self.nconstituents)
            self.feature_names = list(self.record.feature_names)
            log.info(f"Top-tagging cache {self.record.cache_folder}")
        return self.record
