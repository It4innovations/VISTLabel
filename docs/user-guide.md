# VISTLabel user guide

VISTLabel turns any video into a segmentation dataset. You create
objects on a frame, outline them with SAM prompts and the brush, propagate
the masks through the video, review what the model produced and export the
result as a YOLO or COCO dataset.

This guide is for the people who annotate. Installing, configuring and
starting the service is covered by the `README.md` in the repository.

## Workflow at a glance

1. [Create a project](#create-a-project) from an MP4 video, or from an
   existing COCO or YOLO dataset.
2. Open it and [create an object](#create-an-object) for each thing you
   want segmented. The name you type is its class.
3. [Segment it with point prompts](#point-prompts) and accept the
   candidate mask, then fix the edges with the
   [brush and eraser](#brush-and-eraser).
4. [Propagate](#propagation) the masks forwards or backwards through the
   video.
5. [Review](#review) the frames, starting with the ones in the review
   queue.
6. Optionally assign [train / val splits](#splits), then
   [export](#export) the dataset.

## Projects

The **Projects** page lists every project as a card with a thumbnail of the
first frame, its size, frame rate and frame count, and a progress bar:
blue for annotated frames, green for reviewed ones. Type in the search box
to filter the list by name.

Each card has:

- a **status** list (New, In progress, Done, Archived): a label for your
  own bookkeeping. It doesn't change any annotation;
- **Delete**, which removes the whole project once you type its name to
  confirm;
- **Export**, which opens the [export dialog](#export) without opening
  the project;
- **Open**, which opens the [annotation workspace](#annotation-workspace).

### Create a project

Click **Create project**, give it a name and pick the source.

**MP4 video.** Choose the file, then optionally:

- **Resolution**: keep the native size, fit inside 1920×1080, or fit
  inside a custom width and height. The aspect ratio is kept and a video
  is never upscaled.
- **Frames**: keep every frame, keep every *n*th frame (frames 0, n, 2n,
  ...) or resample to a target frame rate, which must not exceed the
  video's own.

The video is re-encoded once at upload when either option changes it, and
the card notes what it was converted from.

**COCO or YOLO dataset (.zip).** The images become the project's frames,
one image per frame, in filename order, played back at the frame rate you
set. Masks and polygons become masks; box-only annotations become
rectangle masks you can refine with SAM. A dataset with both train and val
keeps that split as [split ranges](#splits). Use **Keep every nth image**
to thin a large dataset. Anything that couldn't be imported is listed
after the import.

Uploads show a progress bar and can be cancelled.

## Annotation workspace

The workspace has four areas:

- the **header**: back to Projects, undo and redo, frame navigation and
  playback, the segmentation **Backend**, the project **Status**, and the
  **Auto**, **Runs**, **Export** and **Help** buttons;
- the **tool rail** on the left: positive point, negative point, brush,
  eraser and axis hint;
- the **frame** in the middle, with the action bar and a status line
  under it. The status line reports what each action did, or why it
  couldn't;
- the **side panel** on the right, with the Objects, Review, Queue and
  Splits tabs and the Appearance settings.

Everything you change is saved as you go. There is no Save button.

### Frames and playback

Use the arrow buttons or <kbd>←</kbd> / <kbd>→</kbd> to step through
frames, drag the slider, or type a frame number. **Play** (<kbd>Space</kbd>)
plays forwards at ¼× to 2× of the video's frame rate and stops at the last
frame. Clicking the frame, navigating or using an editing shortcut stops
playback first, so an edit always lands on the frame you see.

VISTLabel remembers the last frame and the selected object per project and
reopens there.

### Appearance

- **Mask opacity** sets how strongly masks are drawn over the frame.
- **Show bounding boxes** draws each object's box and class, exactly as a
  bounding-box export would write it.
- **Zoom** scales the frame. <kbd>Ctrl</kbd> + mouse wheel over the frame
  zooms around the pointer.
- **Brush / eraser radius**. <kbd>Shift</kbd> + mouse wheel over the frame
  changes it too.

These settings only change what you see. Nothing in them is saved into the
annotations.

## Objects and classes

An **object** is one instance tracked through the video, such as one
pipette. Its **class** is the name it's exported under, such as
`pipette`. Several objects can share a class.

### Create an object

In the **Objects** tab, type a name and press **+ New** or
<kbd>Enter</kbd>. Classes you've already used are suggested as you type.
The new object is selected straight away, so you can start
[segmenting](#point-prompts) it.

The list shows the objects on the current frame, the selected object, and
new objects that don't have a mask yet. **Show N not on this frame** under
the list shows the rest. Click an object to select it. Each row shows the
object's state on this frame: `visible`, `partially_occluded` (the mask has
several separate parts), `fully_occluded` (empty), `inactive` or `hidden`.
Hover over the name to see why a frame was flagged for review.

### Per-frame controls

These act on one object on the current frame only:

- **Hide / Show** (<kbd>H</kbd>) stops drawing the mask on this frame.
  This is view only: the mask is still saved and exported.
- **Deactivate / Activate** (<kbd>D</kbd>) takes the object off this
  frame and out of its export, for example when it has left the view. The
  mask is kept, so **Activate** restores it exactly. Propagation leaves an
  inactive object inactive.

### Object-wide controls

These act on the object on every frame:

- **Class** turns the name into a text box. <kbd>Enter</kbd> moves the
  object to the typed class on every frame and <kbd>Esc</kbd> cancels. A
  class no other object uses leaves the class list.
- **Lock / Unlock**: a locked object can't be deleted, neither on its own
  nor with its class. You can still edit, hide and deactivate it.
- **Delete** removes the object and its masks from the whole project after
  you confirm. [Undo](#undo-and-redo) brings it back.

### Class order

The **Class order** section lists the project's classes. The number in
front of each class is its class id in YOLO and COCO exports.

- **↑ / ↓** move a class, which renumbers it in every later export.
- Edit the name and press <kbd>Enter</kbd> to rename the class for all its
  objects. Renaming it to another class's name offers to merge the two.
- **Merge** moves every object of the class into another class you pick.
  The other class keeps its place in the order.
- **✕** deletes the class with all its objects and their masks after you
  confirm. Undo brings them back. A class with a locked object can't be
  deleted.

## Segmenting with SAM

### Choose a backend

The **Backend** list in the header picks the model behind the point
prompts, text search and propagation:

| Backend | Point prompts | Text search | Propagation | Auto |
| --- | --- | --- | --- | --- |
| SAM 2.1 | yes | no | yes | no |
| SAM 3 | yes | yes | yes | no |
| YOLO + SAM | no | no | yes | yes |
| Fake Backend | yes | no | yes | no |

SAM 3 keeps only one of its two models in GPU memory, one for text search
and one for points and propagation. The first request after you switch
between them reloads the model, so it takes noticeably longer.

The Fake Backend draws simple synthetic masks. It's for testing VISTLabel
without a GPU, not for real annotation. A backend that isn't installed or
configured is shown disabled, with the reason. Buttons that the selected
backend can't serve are disabled too, and their tooltip says why. Ask your
administrator to enable a backend.

### Point prompts

1. Select an object.
2. Pick **Positive point** (<kbd>P</kbd> or <kbd>1</kbd>) and click on
   the object. The model returns a **candidate** mask, drawn over the
   frame, and the status line shows its part count and score.
3. Add more positive points to grow the mask, or **Negative point**
   (<kbd>N</kbd> or <kbd>2</kbd>) clicks on areas that must not be
   included. Each click refines the same candidate.
4. Click **Accept candidate** to save it as the object's mask on this
   frame, or **Clear points** to start over.

Nothing is saved until you accept.

### Several visible parts

When something covers the middle of an object, such as fingers over a
pipette, the object shows up as separate visible parts. Segment and
accept the first part, then hold <kbd>Shift</kbd> while clicking a point
on the next one: the candidate is the saved mask plus that part, instead
of a new mask replacing it. Accept it, and repeat for each further part.
The parts stay one object, and the area between them stays unlabelled.

Put a positive point on **every** visible part before you propagate. A
single point that ends up under the occluder is often lost for the rest of
the video, while one point per part tracks through the occlusion.

### Find objects by description

With **SAM 3** selected, type a description such as `pipette` into
**Find objects by description** and click **Find in this frame** (or press
<kbd>Enter</kbd>). Pick one of the matching instances, confirm or correct
its class, and click **Accept as new object**. Changing frames or backends
discards the matches.

### Automatic pre-annotation

**Auto** in the header runs a detector (YOLO + SAM) on the frames you list,
for example `0-99, 150`. Leave the field empty for the current frame only.
It creates objects for what it finds. Detection runs on every frame you
list, so a wide range takes a while. The job shows its progress and can be
cancelled.

A detection whose class isn't one of this project's classes is skipped. If
a run finds nothing, the class names probably don't match yet: create one
object of each class by hand first.

## Brush and eraser

Select an object, then pick **Brush** (<kbd>B</kbd> or <kbd>3</kbd>) to
paint onto its mask or **Eraser** (<kbd>E</kbd> or <kbd>4</kbd>) to remove
from it. With the brush, a right-drag erases, so you don't have to switch
tools. The ring under the pointer shows the radius (dashed for the
eraser). Change the radius in Appearance or with <kbd>Shift</kbd> + mouse
wheel.

Each stroke is saved when you release the mouse. A brush-edited mask
counts as manually edited: propagation won't overwrite it, and the
**Manually edited only** export option picks exactly those frames.

## Undo and redo

**Undo** (<kbd>Ctrl</kbd>+<kbd>Z</kbd>) reverts the last brush or eraser
stroke, and also brings back a deleted object, or a class deleted with ✕,
with all its masks on every frame. **Redo** (<kbd>Ctrl</kbd>+<kbd>Y</kbd>)
re-applies it.

Brush and eraser strokes can be undone while you stay on the same object
and frame. Deletes stay undoable while the page is open, across frame
changes; strokes made after a delete are undone first. Reloading the page
clears the undo history.

## Propagation

Propagation carries masks from the current frame through the rest of the
video with the selected backend (SAM 2.1, SAM 3 or YOLO + SAM).

1. Go to a frame where your objects have good, accepted masks.
2. In the action bar under **Propagate**, choose **Forward** or
   **Backward** and the frame to stop at.
3. Click **Run**.

Every object that has a mask on the current frame is propagated, not only
the selected one. The status shows the progress, and **Cancel** stops the
run. Frames already written are kept.

Propagation never overwrites a mask you edited with the brush or eraser,
or an object you deactivated on a frame. Frames where the result looks
doubtful, such as an object disappearing or reappearing, its area jumping,
or its part count changing, are flagged for [review](#review-queue).

With **Notify** ticked, VISTLabel plays a short sound when a run finishes,
fails or pauses, and shows a browser notification when the tab isn't in
front. Browser notifications need `https` or `localhost`.

### Propagation runs

**Runs** in the header lists every propagation run of the project,
including ones started in another browser or before a restart. A run can
be paused, resumed or cancelled there. A run interrupted by a restart is
never resumed automatically: it is listed as *interrupted* and continues
from its last saved frame only when you resume it.

## Occlusion and axis hints

**Occluder.** When one object covers another (a hand over a pipette),
select the covered object, pick the covering one in the **Occluder** list
and click **Mark**. This records which object occludes which on this
frame. It's extra information for analysis and doesn't change any mask or
export.

**Axis hint.** For a long, thin object, pick **Axis hint**
(<kbd>A</kbd> or <kbd>5</kbd>) and click two or more points along the
object's main axis, then **Save axis hint**. The hint belongs to the
selected object. **Delete axis hint** removes it.

## Review

### Frame review status

Every frame has a review status: **Unreviewed**, **Needs review** (amber)
or **Reviewed** (green). The **Review status** list at the right end of the
action bar shows the current frame's status and saves a new one as soon as
you pick it. <kbd>R</kbd> marks the current frame reviewed. The **Review**
tab does the same and also takes an optional comment.

**Mark all annotated frames reviewed** in the Review tab marks every
annotated frame that isn't reviewed yet. It tells you how many frames that
is, including how many are in the queue, before anything changes.

### Review queue

The **Queue** tab lists the frames flagged for review, with the reasons
and any review comment. The number on the tab is how many there are.
Click an entry to jump to that frame and select the object. Once it looks
right, click **Solved** to mark the frame reviewed and drop it from the
queue. **Refresh** reloads the list.

### Delete frame

**Delete frame** in the action bar leaves the current frame out of every
export and out of the review queue, for example a blurred or duplicated
frame. The video frame and its masks are kept and the frame shows a
*Deleted* badge. **Restore frame** brings it back.

## Splits

The **Splits** tab assigns frame ranges to **train**, **val** or
**excluded**. Ranges are inclusive at both ends. Add as many as you need,
then click **Save splits**.

- With **no ranges at all**, an export puts the last 20% of the exported
  frames, in frame order, into val and the rest into train. A contiguous
  block keeps near-identical neighbouring frames from ending up on both
  sides.
- Once **any range is saved**, only frames inside a train or val range are
  exported. Frames that no range covers are left out, and the export
  report says how many.

The export report also warns when two neighbouring frames land in
different splits.

## Export

**Export** in the header, or on the project card, opens the export dialog.

**Which frames** decides what goes in. The first two options show live
counts.

| Option | Exports |
| --- | --- |
| Reviewed frames only | frames marked Reviewed |
| All annotated frames | every frame with at least one mask, reviewed or not |
| Manually edited only | frames with a brush- or eraser-edited mask |
| Every nth reviewed / annotated | every *n*th of those frames |
| Selected intervals | frames in the ranges you type, e.g. `0-99, 150-200` |
| Every frame that has annotation data | every frame the project ever touched; tick the confirmation |

Deleted frames and deactivated objects are never exported.

The other options:

- **Annotations**: segmentation masks, or bounding boxes (the tight box
  around each mask).
- **Dataset format**: Ultralytics YOLO, or COCO in the layout RF-DETR
  trains from (`train/` and `valid/`, each with an
  `_annotations.coco.json`).
- **Image format**: JPG or PNG.
- **Negative frames**: the fraction of never-annotated frames to add with
  empty labels (1 = all of them), spread evenly through the video. They
  teach the model what *not* to detect.
- **Multi-part masks in YOLO**: a YOLO polygon can't describe a mask with
  separate parts, so choose whether such a mask fails the export, is
  bridged into one outline, or is split into separate instances. COCO
  always keeps the exact shape.
- **Always include exact COCO annotations** adds COCO annotations next to a
  YOLO export.

**Project default** uses the setting from the server configuration. Click
**Start export** and wait for the progress bar. **Previous exports** lists
earlier exports with their settings and a **Download ZIP** link.

## System status and theme

**System status** on the Projects page shows the environment checks
(storage, FFmpeg, GPU), which model backends are available and why any
aren't, and basic counters. Mention it when you report a problem. The
sun/moon button in the header switches between the dark and light theme,
which your browser remembers.

## Keyboard shortcuts

Letter and number shortcuts don't fire while you type in a text field, and
they leave browser shortcuts such as <kbd>Ctrl</kbd>+<kbd>R</kbd> alone.
The number keys go by physical key, so they also work on a Czech keyboard
(<kbd>+</kbd> <kbd>ě</kbd> <kbd>š</kbd> <kbd>č</kbd> <kbd>ř</kbd>). Tool
shortcuts keep working right after you use a slider or checkbox; only the
arrow keys move the slider then.

<!-- shortcuts:begin -->
| Keys | Where | Action |
| --- | --- | --- |
| `Space` | workspace | Play / pause the frames (also while the frame slider or number box has focus) |
| `←` | workspace | Previous frame |
| `→` | workspace | Next frame |
| `Ctrl+Z` | workspace | Undo the last brush or eraser stroke, or bring back a deleted object or class |
| `Ctrl+Y` | workspace | Redo |
| `P / 1` | workspace | Positive point tool |
| `N / 2` | workspace | Negative point tool |
| `B / 3` | workspace | Brush tool |
| `E / 4` | workspace | Eraser tool |
| `A / 5` | workspace | Axis hint tool |
| `H` | workspace | Hide / show the selected object's mask on this frame (view only, still exported) |
| `D` | workspace | Deactivate / activate the selected object on this frame (leaves it out of the export) |
| `R` | workspace | Mark the current frame reviewed |
| `Shift+Click` | canvas | With a point tool: add a separate visible part to the selected object's mask |
| `Right-drag` | canvas | With the brush: erase instead of paint |
| `Ctrl+Wheel` | canvas | Zoom around the pointer |
| `Shift+Wheel` | canvas | Change the brush / eraser radius |
| `Enter` | text field | In the object name box: create the object. In a class name box: rename the class. In the description box: find in this frame |
| `Esc` | text field | Cancel a class rename, class change or merge |
| `Esc` | dialogs | Close the open dialog |
<!-- shortcuts:end -->
