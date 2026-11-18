# Human Generator 3D (HumGen3D) in the pipeline

Human Generator 4.0.14 (Ultimate pack, bought by the user) lives in `tools/Human/` (never pushed: the add-on
code is GPL, but the 3D assets and textures are licensed to the buyer only and must not be redistributed).
It gives realistic humans on **one shared topology**, rigged, with shape-key sliders (body, face, age, height),
skins, eyes, strand hair, clothes, shoes, poses and expressions.

## Install (once, or after a Blender upgrade)

```
tools/blender-5.2.3-windows-x64/blender.exe -b -P pipeline/humgen/install.py -- <abs path tools/Human/HumGen3D> <abs path tools/Human/content> tools/Human/*.hgpack
```

Copies the add-on into the portable Blender, applies `compat.py` (the add-on was written for Blender 3.x; five
fixes make it run in 5.2), sets its content folder, unpacks the packs (2.7 GB). Ends with `HUMGEN_INSTALL_OK`.

## Make a human

```
blender -b -P pipeline/humgen/make_human.py -- <out_dir> --gender female --preset Lisa --outfit Smart_Casual --footwear Leather_Sneakers --hair "Long\Ponytail.json" --pose Standing_looking_at_watch [--age 60]
blender -b -P pipeline/humgen/make_human.py -- <out_dir> --gender male --list   # what's available
```

Writes `human.blend`, front / three-quarter / side / face renders (Eevee) and `human.json`. About 25 s.

Content (Base packs): 41 female and the male presets, 24 female outfits, 8 female shoes, 36 female hairstyles,
52 poses, expressions.

## Children (pipeline/humgen/child/)

Human Generator only makes adults (age 20-70). Children are added as HG livekeys made from MPFB's CC0 child
shapes, carried onto HG's mesh by deformation transfer (HG's rig, eyes, teeth and clothes follow the body):

```
blender -b -P pipeline/humgen/child/export_hg.py -- hg_female.npz --gender female
blender -b -P pipeline/humgen/child/export_mpfb.py -- hg_female.npz mpfb_adult.npz --gender female --age 0.5
blender -b -P pipeline/humgen/child/export_mpfb.py -- hg_female.npz mpfb_child.npz --gender female --age 0.094
blender -b -P pipeline/humgen/child/transfer.py -- hg_female.npz mpfb_adult.npz mpfb_child.npz out.npz --height 1.15 --livekey <content>/livekeys/special/age_child/female_child_6.npz
blender -b -P pipeline/humgen/make_human.py -- <out> --preset Lilly --reset-body --child-look --face-scale 0.4 --key child_6=1 --skin "4K\Female 08" --outfit "Extra Outfits Pack\Dress" --footwear Suede_Dress_Shoe --hair "Medium Center Part" --pose Standing_waving
```

Gown: add `--gown "#7A1048"` (with a dress outfit). `gown.py` keeps the dress down to the hips, rebuilds the
skirt as even rings flaring to the floor with soft folds, weights it to the hips bone, and gives it a satin
material. `--gown-flare` (default 1.0) sets how wide the hem is.

MPFB age macro: 0 = 1 year, 0.1875 = 11, 0.5 = 25 (linear between). MakeHuman's children run short, so
`--height` sets the real height. Livekey files are named `female_<name>.npz` (HG strips the gender prefix).
`overlay.py` draws point clouds over each other to check alignment. Clothes use `smooth_deform.py` (a compat
patch): HG's own refit tore garments on a body change this large.

## A face from a photo onto an HG body

From face_from_image.py's head (tests/face/<name>/head.blend):

```
blender -b -P pipeline/humgen/face_capture.py -- tests/face/flow_woman/head.blend tests/humgen/face_flow_woman --name flow_woman
blender -b -P pipeline/humgen/face_bake.py -- tests/face/flow_woman/head.blend tests/humgen/face_flow_woman --name flow_woman
tools/pose-env/Scripts/python.exe -I pipeline/humgen/face_blend.py tests/humgen/face_flow_woman "<content>/textures/female/Default 4K/Female 05.png" "<content>/textures/female/Default 4K/face_flow_woman.png" --tone detail
blender -b -P pipeline/humgen/make_human.py -- <out> --preset Emma --reset-body --height-cm 166 --face-scale 0 --key face_flow_woman=1 --skin face_flow_woman --hair "Long\Bob Long" --brows none --iris "#3A2618"
```

1. face_capture: the face shape as an HG face slider (`face_<name>`), from a neutral HG face; clay side by side.
2. face_bake: the photo colour, photo coverage and hair mask baked onto HG's skin UVs (Cycles).
3. face_blend: face mask (coverage, no hair, one piece, holes filled, feathered); `--tone detail` keeps the
   photo's detail on HG's broad skin colour (no patch). The result is an HG skin texture option.
4. make_human: the face slider on (preset face sliders off), the new skin, the photo's own brows.

### All of it in one command

```
python pipeline/humgen/photo_character.py tests/face/flow_woman tests/humgen/photo_char_flow_woman --name flow_woman --body tests/body/flow_woman
```

Runs the steps above, then takes the values from measurements instead of hand-picking them: the HG skin closest
in colour to the photo's, height from the body run, hair colour from the face run, and three rounds of render ->
measure (MediaPipe, `measure_eyes.py`, `measure_skin.py`) -> correct for iris colour, eye-white colour and skin
saturation. The comparison shot (`face_shot.py`) uses the Standard view (AgX washes out colour) and an exposure
that follows the photo's; eyes are compared relative to the skin so exposure doesn't matter. Still chosen by flags:
hairstyle, outfit, shoes, body preset. Writes values.json and compare.jpg (about 4 minutes).

## Known Blender 5.2 issues

Fixed in `compat.py`: `eevee.use_gtao`, bpy collection subclassing (crashed Blender), bone groups, bone
selection. Not yet exercised: expressions, face/body sliders by script, baking, LODs, export (`human/process/`).
When one fails, add a patch to `compat.py` (exact old and new text, and why) rather than editing the add-on.
