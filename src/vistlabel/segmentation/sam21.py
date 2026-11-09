# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""SAM 2.1 backend adapter.

Uses the Hugging Face ``transformers`` port (``Sam2VideoModel`` /
``Sam2VideoProcessor``) rather than a `git clone` of
`facebookresearch/sam2`: it is a normal pip dependency (pinned in
`pyproject.toml`'s `gpu` extra) instead of a "clone + `pip install -e .`"
step, and checkpoints come from the Hugging Face Hub via
``scripts/download_models.py`` -- pinned to an exact commit revision, never
fetched at request time or from application startup.

The whole video is decoded once into a SAM2 video session at
``open_video`` time (same full-decode-pass tradeoff already accepted for
the frame-timestamp index, not a new scaling cost) so a
user can click on *any* frame first, not only frame 0. Decoding and
preprocessing happen in bounded chunks feeding the session's streaming
``add_new_frame`` API: at no point does the full-resolution video exist
in memory as one frame list or one stacked tensor -- only the
model-input-resolution frames (~1024x1024, session dtype) accumulate on
the storage device. Passing ``video=`` to ``init_video_session`` instead
would stack every full-resolution frame into a single device tensor
(~270 GB for a 4K/2700-frame recording) and OOM.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from vistlabel.persistence.rle import count_connected_components
from vistlabel.segmentation.base import (
    BackendCapability,
    BackendCapabilityError,
    BackendDescriptor,
    BackendUnavailableError,
    ConceptProposal,
    ExemplarPrompt,
    MaskPrediction,
    PointPrompt,
)
from vistlabel.segmentation.checkpoints import hash_checkpoint_file
from vistlabel.segmentation.torch_utils import (
    clear_object_memory,
    iter_preprocessed_video_chunks,
    resolve_device,
    resolve_dtype,
)
from vistlabel.settings import RuntimeSettings, Sam21ModelSettings
from vistlabel.video.frames import VideoFrameError

if TYPE_CHECKING:
    import torch

_CAPABILITIES = frozenset(
    {
        BackendCapability.POINT_PROMPTS,
        BackendCapability.BOX_PROMPTS,
        BackendCapability.MASK_PROMPTS,
        BackendCapability.VIDEO_PROPAGATION,
        BackendCapability.MULTI_OBJECT,
        BackendCapability.MULTIPART_VISIBLE_MASK,
    }
)

_REQUIRED_CHECKPOINT_FILES = ("config.json", "model.safetensors")


@dataclass
class _Session:
    session_id: str
    inference_session: Any  # transformers.Sam2VideoInferenceSession
    height: int
    width: int
    obj_id_by_object_id: dict[str, int] = field(default_factory=dict)
    _next_obj_id: int = 1

    def get_or_create_obj_id(self, object_id: str) -> int:
        if object_id not in self.obj_id_by_object_id:
            self.obj_id_by_object_id[object_id] = self._next_obj_id
            self._next_obj_id += 1
        return self.obj_id_by_object_id[object_id]


class Sam21Backend:
    backend_id = "sam21"

    def __init__(self, settings: Sam21ModelSettings, runtime_settings: RuntimeSettings) -> None:
        self._settings = settings
        self._runtime_settings = runtime_settings
        self._model: Any = None
        self._processor: Any = None
        self._device: str | None = None
        self._dtype: torch.dtype | None = None
        self._sessions: dict[str, _Session] = {}

    # -- availability -----------------------------------------------------

    def _checkpoint_dir_ready(self) -> bool:
        checkpoint = self._settings.checkpoint
        if checkpoint is None or not checkpoint.is_dir():
            return False
        return all((checkpoint / name).is_file() for name in _REQUIRED_CHECKPOINT_FILES)

    def _unavailable_reason(self) -> str | None:
        if not self._settings.enabled:
            return "disabled in configuration (models.sam21.enabled: false)"
        if self._settings.checkpoint is None:
            return "no checkpoint configured (models.sam21.checkpoint)"
        if not self._checkpoint_dir_ready():
            return (
                f"checkpoint directory missing {_REQUIRED_CHECKPOINT_FILES} under "
                f"{self._settings.checkpoint} -- run scripts/download_models.py"
            )
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError as exc:
            return f"'gpu' extra not installed ({exc}); pip install -e '.[gpu]'"
        return None

    def descriptor(self) -> BackendDescriptor:
        reason = self._unavailable_reason()
        checkpoint_hash = (
            hash_checkpoint_file(self._settings.checkpoint / "model.safetensors")
            if self._checkpoint_dir_ready() and self._settings.checkpoint is not None
            else None
        )
        return BackendDescriptor(
            backend_id=self.backend_id,
            display_name="SAM 2.1",
            capabilities=_CAPABILITIES,
            available=reason is None,
            unavailable_reason=reason,
            model_name=self._settings.variant,
            checkpoint_sha256=checkpoint_hash,
            code_revision=self._settings.code_revision,
        )

    @property
    def device(self) -> str | None:
        """Resolved device (e.g. ``"cuda"``), set once the model has
        actually been loaded; ``None`` before that. Not part of the
        Protocol -- used by diagnostics reporting."""
        return self._device

    @property
    def dtype(self) -> str | None:
        return str(self._dtype) if self._dtype is not None else None

    def _require_available(self) -> None:
        reason = self._unavailable_reason()
        if reason is not None:
            raise BackendUnavailableError(self.backend_id, reason)

    def _ensure_loaded(self) -> None:
        self._require_available()
        if self._model is not None:
            return
        from transformers import Sam2VideoModel, Sam2VideoProcessor

        device = resolve_device(self._runtime_settings.cuda_device)
        dtype = resolve_dtype(self._runtime_settings.default_dtype)
        checkpoint = str(self._settings.checkpoint)
        model = Sam2VideoModel.from_pretrained(
            checkpoint, local_files_only=True, dtype=dtype
        ).to(device)  # type: ignore[arg-type]
        model.eval()
        processor = Sam2VideoProcessor.from_pretrained(checkpoint, local_files_only=True)
        self._model = model
        self._processor = processor
        self._device = device
        self._dtype = dtype

    def _get_session(self, session_id: str) -> _Session:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise KeyError(f"unknown or already-closed session: {session_id}") from exc

    # -- protocol -----------------------------------------------------------

    def open_video(self, project_id: str, video_path: Path) -> str:
        self._ensure_loaded()
        assert self._device is not None  # set by _ensure_loaded
        # Streaming session (no video=): frames are added below in bounded
        # chunks; video= would stack the whole full-resolution video into
        # one device tensor and OOM on long/4K recordings.
        inference_session = self._processor.init_video_session(
            inference_device=self._device, dtype=self._dtype
        )
        height = width = 0
        frame_count = 0
        for pixel_values, height, width in iter_preprocessed_video_chunks(  # noqa: B007 - height/width read after the loop
            self._processor.video_processor, video_path, self._device
        ):
            for frame in pixel_values:
                inference_session.add_new_frame(frame, frame_count)
                frame_count += 1
        if frame_count == 0:
            raise VideoFrameError("no frames could be decoded from this video")
        inference_session.video_height = height
        inference_session.video_width = width
        session_id = uuid.uuid4().hex
        self._sessions[session_id] = _Session(
            session_id=session_id,
            inference_session=inference_session,
            height=height,
            width=width,
        )
        return session_id

    def close_video(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is None:
            return
        del session.inference_session
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _extract_prediction(
        self, session: _Session, output: Any, object_id: str, obj_id: int, frame_index: int
    ) -> MaskPrediction:
        import torch

        index = list(output.object_ids).index(obj_id)
        masks = self._processor.post_process_masks(
            [output.pred_masks],
            original_sizes=[[session.height, session.width]],
            binarize=True,
        )[0]
        mask = masks[index, 0].detach().cpu().numpy().astype(bool)
        score = None
        if getattr(output, "object_score_logits", None) is not None:
            score = float(torch.sigmoid(output.object_score_logits[index]).item())
        return MaskPrediction(
            object_id=object_id,
            frame_index=frame_index,
            visible_mask=mask,
            score=score,
            component_count=count_connected_components(mask),
            backend_metadata={
                "backend": self.backend_id,
                "model_name": self._settings.variant,
            },
        )

    def segment_interactive(
        self,
        session_id: str,
        frame_index: int,
        object_id: str,
        points: list[PointPrompt],
        box_xyxy: tuple[float, float, float, float] | None,
        previous_mask: np.ndarray | None,
    ) -> MaskPrediction:
        self._require_available()
        session = self._get_session(session_id)
        obj_id = session.get_or_create_obj_id(object_id)

        kwargs: dict[str, Any] = {}
        if points:
            kwargs["input_points"] = [[[[point.x, point.y] for point in points]]]
            kwargs["input_labels"] = [[[1 if point.positive else 0 for point in points]]]
        if box_xyxy is not None:
            kwargs["input_boxes"] = [[[list(box_xyxy)]]]
        if previous_mask is not None:
            kwargs["input_masks"] = [previous_mask.astype(np.uint8)]
        if not kwargs:
            raise ValueError(
                "segment_interactive requires at least one of points, box_xyxy, previous_mask"
            )

        self._processor.add_inputs_to_inference_session(
            inference_session=session.inference_session,
            frame_idx=frame_index,
            obj_ids=obj_id,
            **kwargs,
        )
        import torch

        with torch.inference_mode():
            output = self._model(inference_session=session.inference_session, frame_idx=frame_index)
        return self._extract_prediction(session, output, object_id, obj_id, frame_index)

    def propagate(
        self,
        session_id: str,
        object_ids: list[str],
        start_frame: int,
        end_frame: int,
        direction: str,
    ) -> Iterator[MaskPrediction]:
        if direction not in ("forward", "backward"):
            raise ValueError(f"direction must be 'forward' or 'backward', got {direction!r}")
        self._require_available()
        session = self._get_session(session_id)

        for object_id in object_ids:
            if object_id not in session.obj_id_by_object_id:
                raise BackendCapabilityError(
                    self.backend_id,
                    BackendCapability.VIDEO_PROPAGATION,
                    f"object {object_id!r} has no mask on start_frame={start_frame}; "
                    "segment_interactive must be called first",
                )

        import torch

        max_frames = abs(end_frame - start_frame)
        reverse = direction == "backward"
        requested_obj_ids = {session.obj_id_by_object_id[oid]: oid for oid in object_ids}

        with torch.inference_mode():
            for step in self._model.propagate_in_video_iterator(
                session.inference_session,
                start_frame_idx=start_frame,
                max_frame_num_to_track=max_frames,
                reverse=reverse,
            ):
                masks = self._processor.post_process_masks(
                    [step.pred_masks],
                    original_sizes=[[session.height, session.width]],
                    binarize=True,
                )[0]
                step_object_ids = list(step.object_ids)
                for obj_id, object_id in requested_obj_ids.items():
                    if obj_id not in step_object_ids:
                        continue
                    index = step_object_ids.index(obj_id)
                    mask = masks[index, 0].detach().cpu().numpy().astype(bool)
                    score = None
                    if getattr(step, "object_score_logits", None) is not None:
                        score = float(torch.sigmoid(step.object_score_logits[index]).item())
                    yield MaskPrediction(
                        object_id=object_id,
                        frame_index=step.frame_idx,
                        visible_mask=mask,
                        score=score,
                        component_count=count_connected_components(mask),
                        backend_metadata={
                            "backend": self.backend_id,
                            "propagated_from": start_frame,
                        },
                    )

    def replace_object_mask(
        self,
        session_id: str,
        frame_index: int,
        object_id: str,
        visible_mask: np.ndarray,
    ) -> MaskPrediction:
        self._require_available()
        session = self._get_session(session_id)
        obj_id = session.get_or_create_obj_id(object_id)

        self._processor.add_inputs_to_inference_session(
            inference_session=session.inference_session,
            frame_idx=frame_index,
            obj_ids=obj_id,
            input_masks=[visible_mask.astype(np.uint8)],
        )
        import torch

        with torch.inference_mode():
            output = self._model(inference_session=session.inference_session, frame_idx=frame_index)
        return self._extract_prediction(session, output, object_id, obj_id, frame_index)

    def forget_object(self, session_id: str, object_id: str) -> None:
        self._require_available()
        session = self._get_session(session_id)
        clear_object_memory(session.inference_session, session.get_or_create_obj_id(object_id))

    def auto_preannotate(
        self,
        session_id: str,
        frame_indices: list[int],
        requested_class_ids: list[int],
    ) -> Iterator[MaskPrediction]:
        self._require_available()
        raise BackendCapabilityError(
            self.backend_id,
            BackendCapability.CLASS_AWARE_AUTOMATIC_PREANNOTATION,
            "SAM 2.1 does not assign object class labels; "
            "use YoloSamBackend with a configured custom checkpoint",
        )
        yield  # pragma: no cover - makes this a generator to match the Protocol shape

    def find_matching_objects(
        self,
        session_id: str,
        frame_index: int,
        text: str | None,
        exemplar: ExemplarPrompt | None,
    ) -> Iterator[ConceptProposal]:
        raise BackendCapabilityError(
            self.backend_id,
            BackendCapability.TEXT_PROMPTS,
            "SAM 2.1 has no text/exemplar concept-search head; use Sam3Backend",
        )
        yield  # pragma: no cover - makes this a generator to match the Protocol shape

    def find_and_track_matching_objects(
        self,
        session_id: str,
        start_frame: int,
        end_frame: int,
        text: str | None,
        exemplar: ExemplarPrompt | None,
    ) -> Iterator[ConceptProposal]:
        raise BackendCapabilityError(
            self.backend_id,
            BackendCapability.TEXT_PROMPTS,
            "SAM 2.1 has no text/exemplar concept-search head; use Sam3Backend",
        )
        yield  # pragma: no cover - makes this a generator to match the Protocol shape
