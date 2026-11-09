# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Interactive segmentation endpoint: point/box prompts against the active
backend, returning a candidate mask for the browser to render and (once
accepted) PATCH into the frame annotation.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from vistlabel.annotations.png_mask import (
    MaskDecodeError,
    decode_mask_png_base64,
    encode_mask_png_base64,
)
from vistlabel.api.deps import require_frame_index, require_project_with_video
from vistlabel.persistence.paths import source_video_path
from vistlabel.segmentation.base import (
    BackendCapability,
    BackendCapabilityError,
    BackendUnavailableError,
    PointPrompt,
)
from vistlabel.segmentation.registry import UnknownBackendError

router = APIRouter(prefix="/api/v1/projects", tags=["segmentation"])


class PointPromptIn(BaseModel):
    x: float
    y: float
    positive: bool


class SegmentRequest(BaseModel):
    backend_id: str = "fake"
    frame_index: int
    object_id: str
    points: list[PointPromptIn] = []
    box_xyxy: tuple[float, float, float, float] | None = None
    #: Union this in before applying the new prompts -- used for "Add
    #: Visible Component to Selected Object": the client
    #: sends the object's current accumulated mask so a disjoint new point
    #: is added to the same logical object instead of replacing it.
    previous_mask_png_base64: str | None = None


class SegmentResponse(BaseModel):
    frame_index: int
    object_id: str
    visible_mask_png_base64: str
    score: float | None
    component_count: int
    backend_id: str


class ConceptSearchRequest(BaseModel):
    backend_id: str = "sam3"
    frame_index: int = Field(ge=0)
    text: str = Field(min_length=1, max_length=500)

    @field_validator("text")
    @classmethod
    def require_concept(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("a non-empty text concept is required")
        return value


class ConceptProposalResponse(BaseModel):
    proposal_id: str
    frame_index: int
    visible_mask_png_base64: str
    score: float
    suggested_label_name: str
    box_xyxy: tuple[float, float, float, float]


class ConceptSearchResponse(BaseModel):
    frame_index: int
    backend_id: str
    model_name: str
    checkpoint_sha256: str | None
    code_revision: str | None
    proposals: list[ConceptProposalResponse]


@router.post("/{project_id}/segment-concept")
def segment_concept(
    project_id: str, payload: ConceptSearchRequest, request: Request
) -> ConceptSearchResponse:
    """Preview only: labels, objects and annotations change only upon acceptance.

    Like interactive segmentation, this synchronous route runs in FastAPI's
    worker pool, keeping decoding and inference off the event loop.
    """
    _, pdir, timestamps = require_project_with_video(request, project_id)
    require_frame_index(payload.frame_index, timestamps)
    try:
        backend = request.app.state.registry.get(payload.backend_id)
    except UnknownBackendError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    descriptor = backend.descriptor()
    if not descriptor.available:
        raise HTTPException(status_code=409, detail=descriptor.unavailable_reason)
    if BackendCapability.TEXT_PROMPTS not in descriptor.capabilities:
        raise HTTPException(
            status_code=422, detail="selected backend does not support text prompts"
        )
    try:
        session_id = request.app.state.backend_sessions.get_or_open(
            backend, project_id, source_video_path(pdir)
        )
        proposals = [
            ConceptProposalResponse(
                proposal_id=proposal.proposal_id,
                frame_index=proposal.frame_index,
                visible_mask_png_base64=encode_mask_png_base64(proposal.visible_mask),
                score=proposal.score,
                suggested_label_name=proposal.suggested_label_name,
                box_xyxy=proposal.box_xyxy,
            )
            for proposal in backend.find_matching_objects(
                session_id, payload.frame_index, payload.text, None
            )
        ]
    except BackendCapabilityError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except BackendUnavailableError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return ConceptSearchResponse(
        frame_index=payload.frame_index,
        backend_id=descriptor.backend_id,
        model_name=descriptor.model_name,
        checkpoint_sha256=descriptor.checkpoint_sha256,
        code_revision=descriptor.code_revision,
        proposals=proposals,
    )


@router.post("/{project_id}/segment")
def segment(project_id: str, payload: SegmentRequest, request: Request) -> SegmentResponse:
    project, pdir, timestamps = require_project_with_video(request, project_id)
    require_frame_index(payload.frame_index, timestamps)
    assert project.source is not None

    try:
        backend = request.app.state.registry.get(payload.backend_id)
    except UnknownBackendError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    descriptor = backend.descriptor()
    if not descriptor.available:
        raise HTTPException(
            status_code=409,
            detail=(
                f"backend {payload.backend_id!r} is unavailable: {descriptor.unavailable_reason}"
            ),
        )

    session_id = request.app.state.backend_sessions.get_or_open(
        backend, project_id, source_video_path(pdir)
    )

    previous_mask = None
    if payload.previous_mask_png_base64 is not None:
        try:
            previous_mask = decode_mask_png_base64(
                payload.previous_mask_png_base64,
                expected_shape=(project.source.height, project.source.width),
            )
        except MaskDecodeError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    points = [PointPrompt(x=p.x, y=p.y, positive=p.positive) for p in payload.points]
    try:
        prediction = backend.segment_interactive(
            session_id,
            payload.frame_index,
            payload.object_id,
            points,
            payload.box_xyxy,
            previous_mask,
        )
    except BackendCapabilityError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return SegmentResponse(
        frame_index=prediction.frame_index,
        object_id=prediction.object_id,
        visible_mask_png_base64=encode_mask_png_base64(prediction.visible_mask),
        score=prediction.score,
        component_count=prediction.component_count,
        backend_id=backend.descriptor().backend_id,
    )
