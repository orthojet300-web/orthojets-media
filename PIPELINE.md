# Orthojets — automated episode pipeline

This repo lets a fresh Claude session make and publish one Orthojets TikTok episode with no human input.
A scheduled task fires **Monday, Tuesday and Wednesday evenings**. Each run publishes the next episode at **10:00 the following day** (Africa/Tripoli time), so posts land on Tue/Wed/Thu at 10:00.

## Channel rules (from the owner — do not change)
- All on-screen medical content is in **English medical language**. The intro and outro keep their Arabic tagline.
- **Reading time**: after each text block appears, hold it for 1.5–3.5 s, using the `Timeline` holds in `kit/ep_common.py`.
- **Audio**: a calm licensed background track plays quietly under the episode body (-26 LUFS, fade in and out). The intro and outro keep their own sound.
  - Tracks live in the PRIVATE repo `orthojet300-web/orthojets-music`. Clone it next to this repo as `../orthojets-music`.
  - `tools/build_episode.py` picks the track by episode number: track = ((N-1) mod count) + 1.
  - Never copy the MP3s into this public repo: the license forbids redistributing them on their own.
  - If the music repo is unavailable, build without music (`--no-music`) and mention it in the report.
- Brand:
  - Colors: navy `#223F61`, sky `#78AFD8` and white on dark navy. Use green (120,210,170) and coral (236,112,104) only for right-vs-wrong.
  - Fonts: Montserrat ExtraBold/Black. Small text must be at least 27 px.
  - The fonts lack Δ, ≤, ≈ and →. Use `ep_common.rich()` with DejaVuB for Δ, and write the others out in words.
- Layout:
  - Scenes in this order: "ORTHO TIP #NN" hook, context or diagram, rule plus evidence, Tips & Tricks (3 cards), takeaway with the full reference on screen.
  - Body 50–60 s with holds.
  - Keep content within x 60–960 and y 200–1700.
- Anatomy and every number must be accurate. The audience is orthopedic surgeons and medical students.

## Quality bar (from episode 3 on — the owner wants best quality)
- **Richer visuals are mandatory.** Every episode must contain at least one real, animated, anatomically accurate diagram as its centrepiece, not mostly text cards. Examples:
  - a bone or joint drawn as an X-ray, like `kit/femur.py`
  - a cross-section, like the leg compartments in `episodes/ep02_acs`
  - a classification drawn type by type (Salter-Harris I–V, Gartland, Schatzker, Garden)
  - a clinical test shown step by step with a simple limb outline
  - a measurement drawn on the image, like TAD
- **Animate the diagram so it explains something:**
  - structures build in
  - labels point to them
  - the abnormal part is highlighted
  - the measurement is drawn
  - good vs bad is compared side by side
- Text cards support the diagram; they don't replace it.
- Draw the anatomy from known proportions and landmarks, check it on stills, and fix anything wrong before rendering. Simplified schematics are fine if they are labelled and correct.
- **Design review gate (mandatory, before the full render).**
  - Render one still per scene, plus the cover frame. Give them to a separate reviewer agent that has not seen your code.
  - It must PASS each still on all of these:
    - no overlapping or cut-off text
    - all text at least 27 px and readable on a phone
    - no tofu or missing-glyph boxes
    - content inside x 60–960 and y 200–1700
    - anatomy correct and labels pointing to the right structures
    - a centrepiece diagram is present
    - brand colours used
    - the cover frame works as a thumbnail
  - Fix every FAIL and re-check until all pass.
- **Sharpest file.** `tools/build_episode.py` encodes the upload copy with x264 `-preset slow -tune animation` at about **10 Mbps**: roughly 2x the old quality, at 1080x1920, 30 fps, AAC 192k.
  - The bitrate is capped automatically so the file stays under ~90 MB, because GitHub's per-file limit is 100 MB.
  - Do not lower these settings.
  - The encode takes about 5–6 min on top of the render.

## Run steps
1. **Pick the topic.**
   - Open `topics.json`. If an item has status `ready`, publish that one: skip to step 6 using its existing render, or rebuild it if `out/` is empty.
   - Otherwise take the first item with status `todo`.
   - **Keep the queue topped up.** If fewer than 5 items have status `todo`, first append 10 new topics to `topics.json` (status `todo`, the next episode numbers and slugs, and a `source_hint` naming the primary paper(s)).
     - Pick high-yield, practical orthopedic topics that have not been covered yet.
     - Rotate areas: trauma, paediatrics, spine, sports, hand, arthroplasty, oncology, foot & ankle, emergency orthopedics.
     - Each topic needs a clear rule, number, classification or test, backed by a findable primary source.
     - Never repeat a slug or title already in the file.
2. **Research.**
   - Use the PubMed connector to find the primary source(s) named in `source_hint`. Read the abstract and metadata.
   - Collect only figures that appear in the source. Record PMID and DOI.
   - If PubMed is unavailable or the key claim can't be confirmed, set status `blocked` with a reason, push the change, notify the owner, and stop. Never publish unverified medical content.
3. **Write the episode.**
   - Copy `episodes/ep02_acs/acs.py` as the template to `episodes/epNN_slug/script.py`.
   - It must end with `main_cli(TL, SCENES)` so `script.py dur` and `script.py seg` work.
   - The hook scene doubles as the TikTok **cover**. Design it like a thumbnail:
     - The "ORTHO TIP #NN" tag.
     - One big, short headline: a number or 2–4 words.
     - A one-line subtitle saying why it matters.
     - Everything centred between y 400 and 1300, fully on screen by about 2.3 s into the body.
   - Put the sources and verified figures in the module docstring.
4. **Check stills before the full render** (then run the design review gate described under Quality bar).
   - Run `python3 script.py still <t> ...` at one real time per scene, using `TL.real(script_t)`.
   - Look at every still. Fix any overlapping or cut-off text, unreadable small text, wrong anatomy, or missing glyph boxes.
5. **Render.**
   - Run `python3 tools/build_episode.py episodes/epNN_slug/script.py epNN_slug` in the background with a long timeout. It takes about 15–20 min.
   - Poll for `out/epNN_slug_1080p.mp4`. Do not use `pgrep -f` with a pattern that matches your own wait loop.
   - Verify the output:
     - Duration is about 60–70 s.
     - There is an audio stream; max volume is −91 dB across the body and present in the intro and outro.
     - Frames at 2 s, mid-body and the outro look right.
6. **Independent fact check (mandatory).**
   - Spawn a separate agent that has not seen your work.
   - Give it the episode's list of on-screen claims and numbers, plus the PMIDs.
   - It must confirm each claim against PubMed and return PASS or FAIL per claim.
   - On any FAIL, fix the claim and re-render, or set the topic to `blocked`. Never publish a FAIL.
7. **Host the video.** Run `tools/publish_github.sh epNN_slug out/epNN_slug_1080p.mp4`.
   - It force-pushes the MP4 to the orphan branch `media` and prints the public raw URL.
   - GitHub Releases are not permitted for Claude sessions, so a branch is used instead.
   - The `media` branch only ever holds the current video; Metricool copies the file to its own storage when the post is created.
8. **Schedule on TikTok** with the Metricool connector `createScheduledPost`.
   - First call `getScheduledPosts` for the target day. If a TikTok post is already scheduled that day, do not add a second one: move this episode to the next free Tue/Wed/Thu 10:00 slot.
   - Then create the post:
   - blogId `7507086`, timezone `Africa/Tripoli`
   - date: next day 10:00 (`YYYY-MM-DDT10:00:00+02:00`)
   - info:
     - `providers: [{"network":"tiktok"}]`, `autoPublish: true`
     - `media: [<raw URL from step 7>]`
     - `videoCoverMilliseconds`: the frame where the hook is fully on screen. That is about **5000** ms: the intro is 3.0 s, minus the 0.35 s crossfade, plus the moment the hook text has fully appeared, around 2.2–2.5 s into the body.
       - Never use a frame inside the intro (under 3000 ms): it shows the same logo for every episode.
       - Extract that frame with ffmpeg and look at it before scheduling. The tag, big number/title and subtitle must all be visible and sharp.
     - `text`: the English caption — hook line, the rule, 3 ✅ tips, 📚 reference, "Save it & share it with a colleague 👨‍⚕️", then hashtags including #orthopedics #orthopedicsurgery #ortho #medtok #medicalstudent #orthojets plus 3–4 topic tags
     - `tiktokData: {"title":"Ortho Tip #NN · <short title>","privacyOption":"PUBLIC_TO_EVERYONE","disableComment":false,"disableDuet":false,"disableStitch":false,"autoAddMusic":false,"isAigc":false}`
     - `title` is required by Metricool.
     - `isAigc` is false because the videos are motion graphics rendered with code. Set it to true if TikTok's policy requires it for this content.
9. **Record.**
   - In `topics.json`, set the item to `published` with the date, release tag, Metricool post id/plannerUrl and PMIDs.
   - Commit `episodes/`, `topics.json` and any kit fixes. Never commit `out/`.
   - Push.
10. **Report.** Send the owner one short message, in Arabic: the episode title, when it goes live, the Metricool planner link, and the sources (PubMed attribution plus DOI links).

## Failure handling
- Render or upload fails: retry once. If it fails again, leave the topic as `todo`, notify the owner, and stop.
- Metricool rejects the post: notify the owner with the exact error. Do not alter the medical text to work around it.
