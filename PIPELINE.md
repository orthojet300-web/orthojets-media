# Orthojets — automated episode pipeline

This repo lets a fresh Claude session make and publish one Orthojets TikTok episode with no human input.
A scheduled task fires **Monday, Tuesday and Wednesday evenings**. Each run publishes the next episode at **10:00 the following day** (Africa/Tripoli time), so posts land on Tue/Wed/Thu at 10:00.

## Channel rules (from the owner — do not change)
- All on-screen medical content is in **English medical language**. The intro and outro keep their Arabic tagline.
- **Reading time**: after each text block appears, hold it for 1.5–3.5 s, using the `Timeline` holds in `kit/ep_common.py`.
- **Audio**: the episode body is **silent**. The intro and outro keep their own sound (`tools/build_episode.py` does this).
- Brand:
  - Colors: navy `#223F61`, sky `#78AFD8` and white on dark navy. Use green (120,210,170) and coral (236,112,104) only for right-vs-wrong.
  - Fonts: Montserrat ExtraBold/Black. Small text must be at least 27 px.
  - The fonts lack Δ, ≤, ≈ and →. Use `ep_common.rich()` with DejaVuB for Δ, and write the others out in words.
- Layout:
  - Scenes in this order: "ORTHO TIP #NN" hook, context or diagram, rule plus evidence, Tips & Tricks (3 cards), takeaway with the full reference on screen.
  - Body 50–60 s with holds.
  - Keep content within x 60–960 and y 200–1700.
- Anatomy and every number must be accurate. The audience is orthopedic surgeons and medical students.

## Run steps
1. **Pick the topic.**
   - Open `topics.json`. If an item has status `ready`, publish that one: skip to step 6 using its existing render, or rebuild it if `out/` is empty.
   - Otherwise take the first item with status `todo`.
2. **Research.**
   - Use the PubMed connector to find the primary source(s) named in `source_hint`. Read the abstract and metadata.
   - Collect only figures that appear in the source. Record PMID and DOI.
   - If PubMed is unavailable or the key claim can't be confirmed, set status `blocked` with a reason, push the change, notify the owner, and stop. Never publish unverified medical content.
3. **Write the episode.**
   - Copy `episodes/ep02_acs/acs.py` as the template to `episodes/epNN_slug/script.py`.
   - It must end with `main_cli(TL, SCENES)` so `script.py dur` and `script.py seg` work.
   - Put the sources and verified figures in the module docstring.
4. **Check stills before the full render.**
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
7. **Host the video.** Run `tools/publish_github.sh epNN_slug out/epNN_slug_1080p.mp4 "<title>"`. It creates a GitHub Release with the MP4 attached and prints the public download URL.
8. **Schedule on TikTok** with the Metricool connector `createScheduledPost`:
   - blogId `7507086`, timezone `Africa/Tripoli`
   - date: next day 10:00 (`YYYY-MM-DDT10:00:00+02:00`)
   - info:
     - `providers: [{"network":"tiktok"}]`, `autoPublish: true`
     - `media: [<release URL>]`, `videoCoverMilliseconds: 1500` (the hook frame)
     - `text`: the English caption — hook line, the rule, 3 ✅ tips, 📚 reference, "Save it & share it with a colleague 👨‍⚕️", then hashtags including #orthopedics #orthopedicsurgery #ortho #medtok #medicalstudent #orthojets plus 3–4 topic tags
     - `tiktokData: {"privacyOption":"PUBLIC_TO_EVERYONE","disableComment":false,"disableDuet":false,"disableStitch":false,"autoAddMusic":false,"isAigc":false}`
     - `isAigc` is false because the videos are motion graphics rendered with code. Set it to true if TikTok's policy requires it for this content.
9. **Record.**
   - In `topics.json`, set the item to `published` with the date, release tag, Metricool post id/plannerUrl and PMIDs.
   - Commit `episodes/`, `topics.json` and any kit fixes. Never commit `out/`.
   - Push.
10. **Report.** Send the owner one short message, in Arabic: the episode title, when it goes live, the Metricool planner link, and the sources (PubMed attribution plus DOI links).

## Failure handling
- Render or upload fails: retry once. If it fails again, leave the topic as `todo`, notify the owner, and stop.
- Metricool rejects the post: notify the owner with the exact error. Do not alter the medical text to work around it.
