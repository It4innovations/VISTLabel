# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""SAM 3 backend adapter, registered under the ``sam3`` backend id.

**Which model this actually is.** Meta ships SAM 3.1 as a raw
``sam3.1_multiplex.pt`` whose own model card states there is "no Hugging
Face Transformers integration"; loading it would need the
`facebookresearch/sam3` clone, which this project deliberately avoids.
``facebook/sam3`` *is* `transformers`-native, so that is what runs here --
the same kind of version substitution already accepted for SAM 2.1. The
backend id stays ``sam3`` (configuration and stored
``ui_state`` reference it), but ``display_name``/``model_name`` report SAM
3 truthfully rather than claiming 3.1.

``Sam3VideoModel`` provides text-guided concept discovery and tracking.
``Sam3TrackerVideoModel`` is a separate promptable tracker for points,
boxes, masks, and directional propagation. They deliberately use separate
inference sessions, matching the upstream Hugging Face demo.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Iterator
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
from vistlabel.settings import RuntimeSettings, Sam3ModelSettings
from vistlabel.video.frames import VideoFrameError

if TYPE_CHECKING:
    import torch

_CAPABILITIES = frozenset(
    {
        BackendCapability.POINT_PROMPTS,
        BackendCapability.BOX_PROMPTS,
        BackendCapability.MASK_PROMPTS,
        BackendCapability.TEXT_PROMPTS,
        BackendCapability.VIDEO_PROPAGATION,
        BackendCapability.MULTI_OBJECT,
        BackendCapability.MULTIPART_VISIBLE_MASK,
    }
)

_REQUIRED_CHECKPOINT_FILES = ("config.json", "model.safetensors")


@dataclass
class _Session:
    session_id: str
    #: Preprocessed once at open_video to the model's input resolution
    #: (``(num_frames, 3, H', W')`` in CPU memory, session dtype); every
    #: concept search builds a *fresh*
    #: Sam3VideoInferenceSession over this tensor, because add_text_prompt
    #: accumulates on a session and searches must not contaminate each
    #: other. Storing full-resolution PIL frames instead costs ~63 GB for
    #: a 4K/2700-frame recording.
    pixel_values: Any  # torch.Tensor
    height: int
    width: int
    tracker_inference_session: Any | None = None
    obj_id_by_object_id: dict[str, int] = field(default_factory=dict)
    _next_obj_id: int = 1

    def get_or_create_obj_id(self, object_id: str) -> int:
        if object_id not in self.obj_id_by_object_id:
            self.obj_id_by_object_id[object_id] = self._next_obj_id
            self._next_obj_id += 1
        return self.obj_id_by_object_id[object_id]


class Sam3Backend:
    backend_id = "sam3"

    def __init__(self, settings: Sam3ModelSettings, runtime: RuntimeSettings) -> None:
        self._settings = settings
        self._runtime = runtime
        self._model: Any = None
        self._processor: Any = None
        self._tracker_model: Any = None
        self._tracker_processor: Any = None
        self._device: str | None = None
        self._dtype: torch.dtype | None = None
        self._sessions: dict[str, _Session] = {}

    # ------------------------------------------------------------ availability

    def _checkpoint_dir_ready(self) -> bool:
        checkpoint = self._settings.checkpoint
        if checkpoint is None or not checkpoint.is_dir():
            return False
        return all((checkpoint / name).is_file() for name in _REQUIRED_CHECKPOINT_FILES)

    def _unavailable_reason(self) -> str | None:
        if not self._settings.enabled:
            return "disabled in configuration (models.sam3.enabled: false)"
        checkpoint = self._settings.checkpoint
        if checkpoint is None:
            return "no checkpoint configured (models.sam3.checkpoint)"
        if not checkpoint.is_dir():
            return (
                f"checkpoint directory not found: {checkpoint} -- run "
                "`python scripts/download_models.py --model sam3` (gated repo, needs HF_TOKEN)"
            )
        missing = [n for n in _REQUIRED_CHECKPOINT_FILES if not (checkpoint / n).is_file()]
        if missing:
            return f"checkpoint directory {checkpoint} is missing: {', '.join(missing)}"
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError as exc:
            return f"'gpu' extra not installed ({exc}); pip install -e '.[gpu]'"
        return None

    def _require_available(self) -> None:
        reason = self._unavailable_reason()
        if reason is not None:
            raise BackendUnavailableError(self.backend_id, reason)

    def descriptor(self) -> BackendDescriptor:
        reason = self._unavailable_reason()
        checkpoint_hash = (
            hash_checkpoint_file(self._settings.checkpoint / "model.safetensors")
            if self._checkpoint_dir_ready() and self._settings.checkpoint is not None
            else None
        )
        return BackendDescriptor(
            backend_id=self.backend_id,
            display_name="SAM 3",
            capabilities=_CAPABILITIES,
            available=reason is None,
            unavailable_reason=reason,
            model_name=self._settings.variant,
            checkpoint_sha256=checkpoint_hash,
            code_revision=self._settings.code_revision,
        )

    @property
    def device(self) -> str | None:
        """Resolved device, set once the model has been loaded. Not part of
        the Protocol -- used by diagnostics reporting."""
        return self._device

    @property
    def dtype(self) -> str | None:
        return str(self._dtype) if self._dtype is not None else None

    # ------------------------------------------------------------------ loading

    def _ensure_loaded(self) -> None:
        self._require_available()
        if self._model is not None:
            return
        if self._tracker_model is not None:
            self._tracker_model = None
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        from transformers import Sam3VideoModel, Sam3VideoProcessor

        device = resolve_device(self._runtime.cuda_device)
        dtype = resolve_dtype(self._runtime.default_dtype)
        checkpoint = str(self._settings.checkpoint)
        # local_files_only: checkpoints are fetched explicitly by
        # scripts/download_models.py, never at request time.
        model = Sam3VideoModel.from_pretrained(
            checkpoint, local_files_only=True, dtype=dtype
        ).to(device)  # type: ignore[arg-type, unused-ignore]
        model.eval()
        self._model = model
        self._processor = Sam3VideoProcessor.from_pretrained(checkpoint, local_files_only=True)
        self._device = device
        self._dtype = dtype

    def _ensure_tracker_loaded(self) -> None:
        self._require_available()
        if self._tracker_model is not None:
            return
        if self._model is not None:
            self._model = None
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

        device = self._device or resolve_device(self._runtime.cuda_device)
        dtype = self._dtype or resolve_dtype(self._runtime.default_dtype)
        checkpoint = str(self._settings.checkpoint)
        model = Sam3TrackerVideoModel.from_pretrained(
            checkpoint, local_files_only=True, dtype=dtype
        ).to(device)  # type: ignore[arg-type, unused-ignore]
        model.eval()
        self._tracker_model = model
        self._tracker_processor = Sam3TrackerVideoProcessor.from_pretrained(
            checkpoint, local_files_only=True
        )
        self._device = device
        self._dtype = dtype

    # ----------------------------------------------------------------- sessions

    def open_video(self, project_id: str, video_path: Path) -> str:
        self._ensure_loaded()
        import torch

        device = self._device
        assert self._dtype is not None
        assert device is not None
        chunks: list[Any] = []
        height = width = 0
        for pixel_values, height, width in iter_preprocessed_video_chunks(  # noqa: B007 - height/width read after the loop
            self._processor.video_processor, video_path, device
        ):
            # Cast to the session dtype now: chunks arrive in float32 and
            # every fresh per-search session stores this dtype anyway, so
            # keeping float32 here would double the resident footprint.
            # The tracker owns its own frame dictionary. Keep this shared
            # text-search backing tensor in system RAM so it is not a second
            # full-video CUDA allocation. Both SAM 3 session types transfer
            # only the active frame to the inference device.
            chunks.append(pixel_values.to(device="cpu", dtype=self._dtype))
        if not chunks:
            raise VideoFrameError("no frames could be decoded from this video")
        session_id = uuid.uuid4().hex
        self._sessions[session_id] = _Session(
            session_id=session_id,
            pixel_values=torch.cat(chunks),
            height=height,
            width=width,
        )
        return session_id

    def close_video(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is None:
            return
        session.pixel_values = None
        if session.tracker_inference_session is not None:
            session.tracker_inference_session.reset_inference_session()
        session.tracker_inference_session = None
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_session(self, session_id: str) -> _Session:
        session = self._sessions.get(session_id)
        if session is None:
            raise BackendUnavailableError(self.backend_id, f"unknown session: {session_id}")
        return session

    def _get_tracker_session(self, session: _Session) -> Any:
        self._ensure_tracker_loaded()
        if session.tracker_inference_session is not None:
            return session.tracker_inference_session
        device = self._device
        dtype = self._dtype
        assert device is not None
        assert dtype is not None
        inference_session = self._tracker_processor.init_video_session(
            inference_device=device,
            # Persistent per-frame state grows with propagation. Offload it,
            # along with video frames, to system RAM; only active inference
            # tensors and model weights belong in the limited VRAM budget.
            inference_state_device="cpu",
            video_storage_device="cpu",
            dtype=dtype,
        )
        for frame_index, frame in enumerate(session.pixel_values):
            inference_session.add_new_frame(frame, frame_index)
        inference_session.video_height = session.height
        inference_session.video_width = session.width
        session.tracker_inference_session = inference_session
        return inference_session

    # ------------------------------------------------------------ concept search

    def _require_text(self, text: str | None, exemplar: ExemplarPrompt | None) -> str:
        if exemplar is not None:
            raise BackendCapabilityError(
                self.backend_id,
                BackendCapability.EXEMPLAR_PROMPTS,
                "transformers' Sam3VideoProcessor exposes add_text_prompt only; "
                "image exemplars have no entry point in this packaging",
            )
        if not text or not text.strip():
            raise BackendCapabilityError(
                self.backend_id,
                BackendCapability.TEXT_PROMPTS,
                "a non-empty text concept is required",
            )
        return text.strip()

    def _mask_to_full_size(self, raw: Any, session: _Session) -> np.ndarray:
        """SAM 3 returns low-resolution mask logits (e.g. 288x288); bring
        them back to video resolution before anything downstream sees them,
        so no model-specific geometry leaks into stored annotations."""
        import torch
        import torch.nn.functional as F

        tensor = raw if isinstance(raw, torch.Tensor) else torch.as_tensor(raw)
        while tensor.dim() < 4:
            tensor = tensor.unsqueeze(0)
        resized = F.interpolate(
            tensor.float(), size=(session.height, session.width), mode="bilinear",
            align_corners=False,
        )
        return (resized[0, 0] > 0).detach().cpu().numpy().astype(bool)

    @staticmethod
    def _box_of(mask: np.ndarray) -> tuple[float, float, float, float]:
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any() or not cols.any():
            return (0.0, 0.0, 0.0, 0.0)
        y0, y1 = np.where(rows)[0][[0, -1]]
        x0, x1 = np.where(cols)[0][[0, -1]]
        return (float(x0), float(y0), float(x1), float(y1))

    def _search(
        self,
        session: _Session,
        text: str,
        start_frame: int,
        frames_to_track: int,
    ) -> Iterator[ConceptProposal]:
        import torch
        from transformers.models.sam3_video.modeling_sam3_video import (
            Sam3VideoInferenceSession,
        )

        # Built directly from the preprocessed tensor: init_video_session's
        # video= path would re-stack full-resolution frames per search.
        # Same device and dtype as pixel_values, so the session's frame
        # dict is views into the shared tensor, not a copy.
        device = self._device
        dtype = self._dtype
        assert device is not None
        assert dtype is not None
        inference_session = Sam3VideoInferenceSession(
            video=session.pixel_values,
            video_height=session.height,
            video_width=session.width,
            inference_device=device,
            video_storage_device="cpu",
            inference_state_device="cpu",
            dtype=dtype,
        )
        self._processor.add_text_prompt(inference_session, text)

        with torch.inference_mode():
            for output in self._model.propagate_in_video_iterator(
                inference_session,
                start_frame_idx=start_frame,
                max_frame_num_to_track=frames_to_track,
            ):
                for obj_id in list(output.object_ids):
                    mask = self._mask_to_full_size(output.obj_id_to_mask[obj_id], session)
                    if not mask.any():
                        continue
                    score = output.obj_id_to_score.get(obj_id)
                    yield ConceptProposal(
                        # Stable per detected instance across frames, so the
                        # UI can offer "accept this whole track" as one object.
                        proposal_id=f"{text}:{int(obj_id)}",
                        frame_index=int(output.frame_idx),
                        visible_mask=mask,
                        score=float(score) if score is not None else 0.0,
                        suggested_label_name=text,
                        box_xyxy=self._box_of(mask),
                    )

    def find_matching_objects(
        self,
        session_id: str,
        frame_index: int,
        text: str | None,
        exemplar: ExemplarPrompt | None,
    ) -> Iterable[ConceptProposal]:
        self._require_available()
        self._ensure_loaded()
        concept = self._require_text(text, exemplar)
        session = self._get_session(session_id)
        return list(self._search(session, concept, frame_index, frames_to_track=0))

    def find_and_track_matching_objects(
        self,
        session_id: str,
        start_frame: int,
        end_frame: int,
        text: str | None,
        exemplar: ExemplarPrompt | None,
    ) -> Iterable[ConceptProposal]:
        self._require_available()
        self._ensure_loaded()
        concept = self._require_text(text, exemplar)
        session = self._get_session(session_id)
        span = max(0, int(end_frame) - int(start_frame))
        return list(self._search(session, concept, start_frame, frames_to_track=span))

    # -------------------------------------------------------- visual tracking

    def _extract_tracker_prediction(
        self, session: _Session, output: Any, object_id: str, obj_id: int, frame_index: int
    ) -> MaskPrediction:
        import torch

        index = list(output.object_ids).index(obj_id)
        masks = self._tracker_processor.post_process_masks(
            [output.pred_masks], original_sizes=[[session.height, session.width]], binarize=True
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
            backend_metadata={"backend": self.backend_id, "model_name": self._settings.variant},
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
        inference_session = self._get_tracker_session(session)
        obj_id = session.get_or_create_obj_id(object_id)
        kwargs: dict[str, Any] = {}
        if points:
            kwargs["input_points"] = [[[[point.x, point.y] for point in points]]]
            kwargs["input_labels"] = [[[1 if point.positive else 0 for point in points]]]
        if box_xyxy is not None:
            kwargs["input_boxes"] = [[[list(box_xyxy)]]]
        if previous_mask is not None:
            if points or box_xyxy is not None:
                raise ValueError(
                    "SAM 3 tracker cannot combine a mask with points or a box in one request"
                )
            kwargs["input_masks"] = [previous_mask.astype(np.uint8)]
        if not kwargs:
            raise ValueError(
                "segment_interactive requires at least one of points, box_xyxy, previous_mask"
            )
        self._tracker_processor.add_inputs_to_inference_session(
            inference_session=inference_session,
            frame_idx=frame_index,
            obj_ids=obj_id,
            **kwargs,
        )
        import torch

        with torch.inference_mode():
            output = self._tracker_model(inference_session=inference_session, frame_idx=frame_index)
        return self._extract_tracker_prediction(session, output, object_id, obj_id, frame_index)

    def propagate(
        self,
        session_id: str,
        object_ids: list[str],
        start_frame: int,
        end_frame: int,
        direction: str,
    ) -> Iterable[MaskPrediction]:
        if direction not in ("forward", "backward"):
            raise ValueError(f"direction must be 'forward' or 'backward', got {direction!r}")
        self._require_available()
        session = self._get_session(session_id)
        inference_session = self._get_tracker_session(session)
        requested_obj_ids: dict[int, str] = {}
        for object_id in object_ids:
            if object_id not in session.obj_id_by_object_id:
                raise BackendCapabilityError(
                    self.backend_id,
                    BackendCapability.VIDEO_PROPAGATION,
                    f"object {object_id!r} has no seed mask on start_frame={start_frame}",
                )
            requested_obj_ids[session.obj_id_by_object_id[object_id]] = object_id

        import torch

        with torch.inference_mode():
            for output in self._tracker_model.propagate_in_video_iterator(
                inference_session,
                start_frame_idx=start_frame,
                max_frame_num_to_track=abs(end_frame - start_frame),
                reverse=direction == "backward",
            ):
                for obj_id, object_id in requested_obj_ids.items():
                    if obj_id in output.object_ids:
                        yield self._extract_tracker_prediction(
                            session, output, object_id, obj_id, int(output.frame_idx)
                        )

    def replace_object_mask(
        self,
        session_id: str,
        frame_index: int,
        object_id: str,
        visible_mask: np.ndarray,
    ) -> MaskPrediction:
        return self.segment_interactive(
            session_id, frame_index, object_id, [], None, visible_mask
        )

    def forget_object(self, session_id: str, object_id: str) -> None:
        self._require_available()
        session = self._get_session(session_id)
        # No tracker session yet means nothing has been remembered.
        if session.tracker_inference_session is not None:
            clear_object_memory(
                session.tracker_inference_session, session.get_or_create_obj_id(object_id)
            )

    def auto_preannotate(
        self,
        session_id: str,
        frame_indices: list[int],
        requested_class_ids: list[int],
    ) -> Iterable[MaskPrediction]:
        raise BackendCapabilityError(
            self.backend_id,
            BackendCapability.CLASS_AWARE_AUTOMATIC_PREANNOTATION,
            "class-id-driven automatic pre-annotation",
        )
