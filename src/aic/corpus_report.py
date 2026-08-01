"""Corpus readiness and modality-coverage report over the persisted manifests.

A pure, CPU-only analysis: it reads the ingest ``videos`` manifest and the
assembled ``chronicle`` manifest and reports, per video, how many shots carry
each retrieval signal — ASR speech, OCR on-screen text, captions, and the
derived semantic/literal Chronicle documents that feed the two text channels.

Why this exists: the first fixture run (docs/note/11 section 5) showed that
Reciprocal Rank Fusion with equal channel votes is fragile when modality
coverage is grossly uneven across the corpus — a text-rich clip becomes a
"hub" that weakly matches almost every query and floods the top of unrelated
ones, while silent / graphics-only clips are absent from two of three channels
and get buried. That skew is a property of the *data* and is visible in the
manifests before any embedding, labelling, or retrieval. This report surfaces
it so a corpus can be balanced (or the operator warned) at collection time,
the cheapest possible moment to fix it.

Membership in the semantic / literal channels is decided here by the very same
``ChronicleRecord.semantic_text()`` / ``literal_text()`` predicates that
``aic.textstack.job`` uses to admit a shot into each index, so the counts
reported equal the real index membership rather than an approximation. No model
libraries are imported; every number comes from JSONL manifests.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev

from aic.chronicle.jobs import CHRONICLE_MANIFEST, load_chronicle
from aic.chronicle.schema import ChronicleRecord
from aic.config import Config
from aic.ingest.pipeline import VIDEOS_MANIFEST, discover_videos
from aic.manifest import read_manifest

# The two text channels prone to hubness under uneven coverage. The dense
# visual channel embeds every kept keyframe, so its coverage is near-universal
# and it is not a concentration risk; the concern is the Chronicle-derived
# text indexes, whose membership depends on a shot actually having speech /
# on-screen text / a caption.
_TEXT_CHANNELS: tuple[tuple[str, str], ...] = (
    ("semantic", "n_semantic"),
    ("literal", "n_literal"),
)


@dataclass(frozen=True)
class VideoCoverage:
    """Per-video shot counts broken down by which retrieval signal is present."""

    video_id: str
    status: str
    duration_ms: int | None
    has_audio: bool
    n_shots: int
    n_keyframes: int
    n_caption: int
    n_asr: int
    n_ocr: int
    n_semantic: int
    n_literal: int

    def rate(self, count: int) -> float:
        """Fraction of this video's shots covered by ``count`` (0 when no shots)."""
        return count / self.n_shots if self.n_shots else 0.0


@dataclass(frozen=True)
class ChannelConcentration:
    """How one text channel's documents are spread across the corpus.

    ``max_hub_share`` and ``rate_cv`` are plain descriptive statistics, not
    thresholds: they quantify "is one clip dominating this channel" and "how
    uneven are the per-video coverage rates" so a human (or an opt-in warning
    threshold) can judge hub risk. They never gate anything on their own.
    """

    channel: str
    total_docs: int
    total_shots: int
    per_video_docs: dict[str, int]
    per_video_rate: dict[str, float]

    @property
    def coverage(self) -> float:
        """Corpus-wide fraction of shots that contribute a document."""
        return self.total_docs / self.total_shots if self.total_shots else 0.0

    @property
    def hub_video(self) -> str | None:
        """The video holding the most documents in this channel, if any."""
        if not self.per_video_docs or self.total_docs == 0:
            return None
        return max(self.per_video_docs, key=lambda v: self.per_video_docs[v])

    @property
    def max_hub_share(self) -> float:
        """Share of the channel's documents held by its single richest video."""
        if self.total_docs == 0:
            return 0.0
        return max(self.per_video_docs.values()) / self.total_docs

    @property
    def rate_cv(self) -> float:
        """Coefficient of variation of per-video coverage rates.

        0 when rates are identical (perfectly even coverage); grows as the
        rates spread apart. Undefined with fewer than two videos or when every
        rate is zero, reported as 0.0 in those degenerate cases.
        """
        rates = list(self.per_video_rate.values())
        if len(rates) < 2:
            return 0.0
        avg = mean(rates)
        if avg == 0:
            return 0.0
        return pstdev(rates) / avg


@dataclass(frozen=True)
class CorpusReport:
    videos: list[VideoCoverage]
    not_ingested: list[str]
    channels: list[ChannelConcentration]
    has_chronicle: bool


def _video_coverage(
    record: dict, shots: list[ChronicleRecord], has_chronicle: bool
) -> VideoCoverage:
    # When the Chronicle exists its shot rows are the authoritative denominator
    # (the set actually offered to the text indexes); otherwise fall back to the
    # ingest shot count so a corpus that is embedded but not yet chronicled
    # still reports a sensible shot total.
    n_shots = len(shots) if has_chronicle and shots else record.get("n_shots") or 0
    return VideoCoverage(
        video_id=record["video_id"],
        status=record.get("status", "unknown"),
        duration_ms=record.get("duration_ms"),
        has_audio=bool(record.get("has_audio", False)),
        n_shots=n_shots,
        n_keyframes=record.get("n_keyframes", 0),
        n_caption=sum(1 for s in shots if not s.degraded),
        n_asr=sum(1 for s in shots if s.asr_text),
        n_ocr=sum(1 for s in shots if s.ocr),
        n_semantic=sum(1 for s in shots if s.semantic_text()),
        n_literal=sum(1 for s in shots if s.literal_text()),
    )


def build_corpus_report(cfg: Config) -> CorpusReport:
    """Read the manifests and compute the per-video / per-channel coverage report."""
    manifests = cfg.paths.manifests_dir
    video_records = {
        record["video_id"]: record
        for record in read_manifest(manifests / VIDEOS_MANIFEST)
    }

    has_chronicle = (manifests / CHRONICLE_MANIFEST).is_file()
    shots_by_video: dict[str, list[ChronicleRecord]] = {}
    if has_chronicle:
        for record in load_chronicle(cfg):
            shots_by_video.setdefault(record.video_id, []).append(record)

    videos = [
        _video_coverage(video_records[vid], shots_by_video.get(vid, []), has_chronicle)
        for vid in sorted(video_records)
    ]

    not_ingested: list[str] = []
    try:
        present = discover_videos(cfg.paths.videos_dir, cfg.ingest.video_extensions)
        not_ingested = [p.stem for p in present if p.stem not in video_records]
    except FileNotFoundError:
        pass

    channels: list[ChannelConcentration] = []
    if has_chronicle:
        covered = [v for v in videos if v.n_shots]
        for name, attr in _TEXT_CHANNELS:
            per_docs = {v.video_id: getattr(v, attr) for v in covered}
            per_rate = {v.video_id: v.rate(getattr(v, attr)) for v in covered}
            channels.append(
                ChannelConcentration(
                    channel=name,
                    total_docs=sum(per_docs.values()),
                    total_shots=sum(v.n_shots for v in covered),
                    per_video_docs=per_docs,
                    per_video_rate=per_rate,
                )
            )

    return CorpusReport(
        videos=videos,
        not_ingested=sorted(not_ingested),
        channels=channels,
        has_chronicle=has_chronicle,
    )


def coverage_warnings(
    report: CorpusReport,
    min_shot_coverage: float | None = None,
    max_hub_share: float | None = None,
) -> list[str]:
    """Advisory warnings, emitted only for thresholds the caller opts into.

    With both arguments ``None`` (the default) this returns no warnings: the
    report is then a pure description with no baked-in verdict. Passing a value
    makes the caller — not this module — own the threshold, keeping magic
    numbers out of the logic.

    - ``min_shot_coverage``: warn when a text channel's corpus-wide coverage
      (documents / shots) is below the floor, i.e. the channel is too sparse to
      contribute reliable signal.
    - ``max_hub_share``: warn when a single video holds more than this share of
      a channel's documents, i.e. that video is a fusion hub risk.
    """
    warnings: list[str] = []
    for channel in report.channels:
        if min_shot_coverage is not None and channel.coverage < min_shot_coverage:
            warnings.append(
                f"{channel.channel}: coverage {channel.coverage:.2f} is below the "
                f"floor {min_shot_coverage:.2f} — channel is sparse; its votes are "
                f"noise on most queries."
            )
        if max_hub_share is not None and channel.max_hub_share > max_hub_share:
            warnings.append(
                f"{channel.channel}: {channel.hub_video} holds "
                f"{channel.max_hub_share:.2f} of the channel's documents "
                f"(> {max_hub_share:.2f}) — hub risk, it can flood unrelated "
                f"queries under equal-weight RRF."
            )
    return warnings


def _fmt_duration(duration_ms: int | None) -> str:
    if duration_ms is None:
        return "?"
    return f"{duration_ms / 1000:.0f}s"


def render_report(report: CorpusReport) -> str:
    """Render the report as a terminal-friendly table plus channel summaries."""
    lines: list[str] = []
    header = (
        f"{'video':<20} {'status':<7} {'dur':>6} {'aud':>3} {'shots':>5} "
        f"{'kf':>4} {'cap%':>5} {'asr%':>5} {'ocr%':>5} {'sem%':>5} {'lit%':>5}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for v in report.videos:
        lines.append(
            f"{v.video_id:<20.20} {v.status:<7.7} {_fmt_duration(v.duration_ms):>6} "
            f"{'y' if v.has_audio else 'n':>3} {v.n_shots:>5} {v.n_keyframes:>4} "
            f"{v.rate(v.n_caption):>5.2f} {v.rate(v.n_asr):>5.2f} "
            f"{v.rate(v.n_ocr):>5.2f} {v.rate(v.n_semantic):>5.2f} "
            f"{v.rate(v.n_literal):>5.2f}"
        )

    if not report.has_chronicle:
        lines.append("")
        lines.append(
            "chronicle manifest not found: text-channel coverage is unmeasured "
            "(run the ASR/OCR/caption jobs and the assembler). Columns cap/asr/"
            "ocr/sem/lit read 0.00 above."
        )

    for channel in report.channels:
        lines.append("")
        lines.append(
            f"[{channel.channel}] coverage={channel.coverage:.2f} "
            f"({channel.total_docs}/{channel.total_shots} shots)  "
            f"max_hub_share={channel.max_hub_share:.2f}"
            + (f" ({channel.hub_video})" if channel.hub_video else "")
            + f"  rate_cv={channel.rate_cv:.2f}"
        )

    if report.not_ingested:
        lines.append("")
        lines.append(
            f"not ingested ({len(report.not_ingested)}): "
            + ", ".join(report.not_ingested)
        )
    return "\n".join(lines)
